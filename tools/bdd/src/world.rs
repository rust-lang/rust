//! BDD World - holds test state during scenario execution.

use cucumber::World;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// The test world shared across all steps in a scenario.
#[derive(Debug, Default, World)]
pub struct ThingOsWorld {
    /// Target architecture for this test run
    pub arch: String,
    /// QEMU child process
    #[world(skip)]
    pub qemu: Option<Child>,
    /// Accumulated serial output
    pub serial_log: Arc<Mutex<String>>,
    /// Path to QMP socket for QEMU control
    /// Path to QMP connection for step logic (separate from reporter)
    #[world(skip)]
    pub qmp_socket: Option<PathBuf>,
    /// QMP connection for step logic (separate from reporter)
    #[world(skip)]
    pub qmp_control: Option<PathBuf>,
    /// VNC display number (for screenshot capture)
    #[world(skip)]
    pub vnc_display: Option<u16>,
    /// Path to the ISO file created for this scenario
    #[world(skip)]
    pub iso_path: Option<PathBuf>,
    /// Work directory for storing sockets
    #[world(skip)]
    pub work_dir: PathBuf,
}

impl ThingOsWorld {
    /// Boot the OS in QEMU for the given architecture.
    /// This builds a unique ISO with 1920x1080 resolution for this scenario.
    pub async fn boot(&mut self, arch: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.arch = arch.to_string();

        // Generate unique ISO name for this scenario
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let pid = std::process::id();
        let iso_name = format!("thing-os-bdd-{}-{}-{}.iso", arch, pid, nanos);
        let iso_path = PathBuf::from(&iso_name);

        self.work_dir = std::env::temp_dir().join(format!("thingos-bdd-{}-{}", pid, nanos));
        std::fs::create_dir_all(&self.work_dir)?;

        // Get resolution from environment (default 1920x1080 for BDD tests)
        let resolution =
            std::env::var("BDD_RESOLUTION").unwrap_or_else(|_| "1920x1080".to_string());

        // Build ISO using xtask command
        eprintln!(
            "[bdd] Building ISO {} with resolution {}...",
            iso_name, resolution
        );
        let build_status = std::process::Command::new("cargo")
            .args([
                "xtask",
                "iso",
                "--resolution",
                &resolution,
                "--output",
                &iso_name,
            ])
            .env("RUSTFLAGS", "-Awarnings")
            .status()?;

        if !build_status.success() {
            return Err(format!("Failed to build ISO: {}", iso_name).into());
        }

        self.iso_path = Some(iso_path);
        eprintln!("[bdd] ISO built: {}", iso_name);

        let ovmf_code = format!("vendor/ovmf/ovmf-code-{}.fd", arch);
        let ovmf_vars = format!("vendor/ovmf/ovmf-vars-{}.fd", arch);

        // Create unique socket paths for this test run
        let qmp_global_path = PathBuf::from(format!("/tmp/qemu-bdd-global-{}-{}.sock", pid, nanos));
        let qmp_world_path = PathBuf::from(format!("/tmp/qemu-bdd-world-{}-{}.sock", pid, nanos));

        // We store one of them in self for qmp_init helper (though helper needs refactor if I use it for both)
        // Actually, let's just make qmp_init take a path or just inline it.
        // For now, let's store global in qmp_socket (legacy) and handle world manually
        self.qmp_socket = Some(qmp_global_path.clone());

        // Use a random VNC display to avoid conflicts with potential zombies
        let vnc_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let vnc_display = ((vnc_nanos % 5000) + 1000) as u16;
        self.vnc_display = Some(vnc_display);

        let qemu_bin = match arch {
            "x86_64" => "qemu-system-x86_64",
            "aarch64" => "qemu-system-aarch64",
            "riscv64" => "qemu-system-riscv64",
            "loongarch64" => "qemu-system-loongarch64",
            _ => return Err(format!("Unsupported architecture: {}", arch).into()),
        };

        // Build QEMU command with serial output to stdio and QMP control
        let mut cmd = Command::new(qemu_bin);

        // Handle machine type and pflash - riscv64 requires special blockdev syntax
        match arch {
            "x86_64" => {
                cmd.args(["-M", "q35,usb=off,vmport=off,i8042=on"]);
                cmd.args(["-device", "virtio-vga"]);
                cmd.args([
                    "-drive",
                    &format!("if=pflash,unit=0,format=raw,file={},readonly=on", ovmf_code),
                ]);
                cmd.args([
                    "-drive",
                    &format!("if=pflash,unit=1,format=raw,file={}", ovmf_vars),
                ]);
                cmd.args(["-cdrom", &iso_name]);

                cmd.args(["-device", "virtio-net-pci,netdev=n0"]);
                cmd.args(["-netdev", "user,id=n0"]);
            }
            "aarch64" => {
                cmd.args(["-M", "virt"]);
                cmd.args(["-cpu", "cortex-a72"]);
                cmd.args(["-device", "ramfb"]);
                cmd.args(["-device", "qemu-xhci"]);
                cmd.args(["-device", "usb-kbd"]);
                cmd.args(["-device", "usb-mouse"]);
                cmd.args([
                    "-drive",
                    &format!("if=pflash,unit=0,format=raw,file={},readonly=on", ovmf_code),
                ]);
                cmd.args([
                    "-drive",
                    &format!("if=pflash,unit=1,format=raw,file={}", ovmf_vars),
                ]);
                cmd.args(["-cdrom", &iso_name]);
            }
            "riscv64" => {
                // riscv64 virt requires blockdev syntax with machine-level pflash assignment
                // Also uses virtio-blk instead of -cdrom since riscv64 virt doesn't expose cdrom to UEFI properly
                cmd.args([
                    "-blockdev",
                    &format!(
                        "node-name=pflash0,driver=file,read-only=on,filename={}",
                        ovmf_code
                    ),
                ]);
                cmd.args([
                    "-blockdev",
                    &format!("node-name=pflash1,driver=file,filename={}", ovmf_vars),
                ]);
                cmd.args(["-M", "virt,pflash0=pflash0,pflash1=pflash1"]);
                cmd.args(["-cpu", "rv64"]);
                cmd.args(["-device", "ramfb"]);
                cmd.args(["-device", "qemu-xhci"]);
                cmd.args(["-device", "usb-kbd"]);
                cmd.args(["-device", "usb-mouse"]);
                cmd.args([
                    "-drive",
                    &format!("file={},format=raw,if=none,id=drive0,readonly=on", iso_name),
                ]);
                cmd.args(["-device", "virtio-blk-device,drive=drive0"]);
            }
            "loongarch64" => {
                cmd.args(["-M", "virt"]);
                cmd.args(["-cpu", "la464"]);
                cmd.args(["-device", "ramfb"]);
                cmd.args(["-device", "qemu-xhci"]);
                cmd.args(["-device", "usb-kbd"]);
                cmd.args(["-device", "usb-mouse"]);
                cmd.args([
                    "-drive",
                    &format!("if=pflash,unit=0,format=raw,file={},readonly=on", ovmf_code),
                ]);
                cmd.args([
                    "-drive",
                    &format!("if=pflash,unit=1,format=raw,file={}", ovmf_vars),
                ]);
                cmd.args(["-cdrom", &iso_name]);
            }
            _ => {}
        }

        cmd.args([
            "-m",
            "2G",
            "-smp",
            "4",
            // Disable default display, use VNC instead
            "-display",
            "none",
            "-no-shutdown",
            // Serial via UNIX socket to avoid block-buffering delays
            "-chardev",
            &format!(
                "socket,id=char0,path={},server=on,wait=on",
                self.work_dir.join("serial.sock").display()
            ),
            "-serial",
            "chardev:char0",
            // VNC for headless graphics (needed for screenshots)
            "-vnc",
            &format!(":{}", vnc_display),
            // QMP control sockets (TWO of them)
            "-qmp",
            &format!("unix:{},server=on,wait=off", qmp_global_path.display()),
            "-qmp",
            &format!("unix:{},server=on,wait=off", qmp_world_path.display()),
        ]);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped()); // Capture stderr too to see QEMU errors
        cmd.stdin(Stdio::null());

        let mut child = cmd.spawn()?;

        // Spawn a task to read serial output from the UNIX socket and sync to global cache
        let serial_log = self.serial_log.clone();
        let serial_sock_path = self.work_dir.join("serial.sock");

        tokio::spawn(async move {
            use crate::artifacts;
            // Wait briefly for QEMU to create the serial socket
            for _ in 0..50 {
                if serial_sock_path.exists() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }

            match UnixStream::connect(&serial_sock_path).await {
                Ok(mut stream) => {
                    let mut buf = vec![0u8; 4096];
                    let mut last_update = std::time::Instant::now();
                    while let Ok(n) = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await {
                        if n == 0 {
                            break;
                        } // EOF

                        let text = String::from_utf8_lossy(&buf[..n]);
                        let mut log = serial_log.lock().await;

                        log.push_str(&text);

                        // Sync to global cache for reporter access (throttled to avoid QEMU pipe stall)
                        if last_update.elapsed().as_millis() > 50 {
                            artifacts::set_latest_serial(&log).await;
                            last_update = std::time::Instant::now();
                        }
                    }
                    eprintln!(
                        "│  │  │      ⚠️ QEMU serial socket reader loop exited! Did QEMU close?"
                    );
                }
                Err(e) => {
                    eprintln!(
                        "│  │  │      debug: FAILED to connect to QEMU serial socket: {}",
                        e
                    );
                }
            }
        });

        // Also spawn a task to read stderr for QEMU errors
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    eprintln!("[qemu-stderr] {}", line);
                }
            });
        }

        self.qemu = Some(child);

        let global_path = if qmp_global_path.exists() {
            Some(qmp_global_path.clone())
        } else {
            None
        };
        let world_path = if qmp_world_path.exists() {
            Some(qmp_world_path.clone())
        } else {
            None
        };

        crate::artifacts::set_qmp_stream(global_path).await;
        self.qmp_control = world_path;

        Ok(())
    }

    /// Take a screenshot using the world's private QMP connection.
    pub async fn take_screenshot(
        &mut self,
        output_path: &std::path::Path,
    ) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
        use crate::artifacts::qmp::execute_on_stream;

        // Ensure output directory exists
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Get absolute path for QEMU
        let ppm_path = output_path.with_extension("ppm");
        let ppm_abs = std::fs::canonicalize(output_path.parent().ok_or("Invalid output path")?)?
            .join(ppm_path.file_name().ok_or("Invalid PPM path")?);

        let cmd = format!(
            r#"{{"execute": "screendump", "arguments": {{"filename": "{}"}}}}"#,
            ppm_abs.display()
        );

        let path = self.qmp_control.as_ref().ok_or("No world QMP connection")?;
        let mut stream = Self::connect_qmp(path).await.map_err(|e| e.to_string())?;

        let resp = crate::artifacts::qmp::execute_on_stream(&mut stream, &cmd).await?;
        if resp.contains("error") {
            return Err(format!("QMP error: {}", resp).into());
        }

        // Give QEMU a moment to flush
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        if !ppm_path.exists() {
            return Err("Screenshot file not created".into());
        }

        // Convert PPM to PNG
        let png_path = output_path.with_extension("png");
        let img = image::open(&ppm_path)?;
        img.save(&png_path)?;
        let _ = std::fs::remove_file(&ppm_path);

        Ok(png_path)
    }

    pub async fn execute_qmp_control(
        &self,
        cmd: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let path = self.qmp_control.as_ref().ok_or("No world QMP connection")?;
        let mut stream = Self::connect_qmp(path).await?;
        crate::artifacts::qmp::execute_on_stream(&mut stream, cmd).await
    }

    /// Connect to a QMP socket and perform handshake.
    async fn connect_qmp(
        socket_path: &std::path::Path,
    ) -> Result<UnixStream, Box<dyn std::error::Error + Send + Sync>> {
        let mut stream = UnixStream::connect(socket_path).await?;

        // Read greeting
        let mut buf = vec![0u8; 4096];
        let _ = stream.readable().await;
        let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await?;

        // Send qmp_capabilities to enter command mode
        let caps_cmd = r#"{"execute": "qmp_capabilities"}"#;
        stream.write_all(caps_cmd.as_bytes()).await?;
        stream.write_all(b"\n").await?;

        // Read capability response
        let _ = stream.readable().await;
        let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await?;

        Ok(stream)
    }

    /// Wait for a string to appear in the serial log.
    pub async fn wait_for_serial(&self, needle: &str, timeout_secs: f64) -> bool {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f64(timeout_secs);
        let mut last_print = std::time::Instant::now();

        loop {
            {
                let log = self.serial_log.lock().await;
                if last_print.elapsed() > std::time::Duration::from_secs(5) {
                    eprintln!(
                        "│  │  │      debug: waiting for {}. current log len: {}",
                        needle,
                        log.len()
                    );
                    let tail = if log.len() > 200 {
                        &log[log.len() - 200..]
                    } else {
                        &log[..]
                    };
                    eprintln!("│  │  │      debug: tail: {:?}", tail);
                    last_print = std::time::Instant::now();
                }

                if log.contains(needle) {
                    eprintln!("│  │  │      debug: found {} in serial", needle);
                    return true;
                }
            }

            if start.elapsed() > timeout {
                eprintln!("│  │  │      debug: timed out waiting for {}", needle);
                return false;
            }

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    /// Get the current serial log contents.
    pub async fn get_serial_log(&self) -> String {
        self.serial_log.lock().await.clone()
    }

    /// Kill the QEMU process if running and clean up the ISO.
    pub async fn shutdown(&mut self) {
        // Wait a bit to ensure any pending screenshots/logs are captured
        // The user specifically requested to keep QEMU open long enough.
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        if let Some(ref mut child) = self.qemu {
            let _ = child.kill().await;
        }

        // Clean up global QMP stream before removing socket file (although kill closes it)
        crate::artifacts::set_qmp_stream(None).await;

        // Clean up QMP socket
        if let Some(ref socket_path) = self.qmp_socket {
            let _ = std::fs::remove_file(socket_path);
        }

        // Clean up ISO file
        if let Some(ref iso_path) = self.iso_path {
            eprintln!("[bdd] Cleaning up ISO: {}", iso_path.display());
            let _ = std::fs::remove_file(iso_path);
        }
    }
}

// ===== Bag Matcher and Diagnostics Support =====

/// Check if runtime diagnostics mode is enabled.
pub fn diag_enabled() -> bool {
    std::env::var("THINGOS_DIAG").is_ok()
}

/// Required boot signals that must ALL appear (order irrelevant).
/// Each entry is a list of acceptable alternatives - pass if ANY in the group matches.
/// These signals now use [CONTRACT] prefix for critical boot milestones.
pub const REQUIRED_BOOT_SIGNALS: &[&[&str]] = &[
    // Kernel start - now uses CONTRACT level
    &["[CONTRACT]", "thing-os kernel"],
    // SIMD init (replaces legacy paging boundary signal)
    &["[CONTRACT]", "Initializing SIMD"],
    // Memory map / allocator
    &["[CONTRACT]", "Frame allocator initialized"],
    &["[CONTRACT]", "Initializing global allocator"],
    // Tasking bring-up
    &["[CONTRACT]", "Initializing tasking"],
    &["[CONTRACT]", "Scheduler initialized"],
    &["[CONTRACT]", "Entering scheduler loop"],
];

/// Liveness signals - at least one of these must appear.
pub const LIVENESS_SIGNALS: &[&str] = &[
    "A: tick",
    "B: tick",
    "threads_demo",
    "BOOT: heartbeat",
    "BOOT: ready",
];

impl ThingOsWorld {
    /// Wait until all required signals are found in the log (unordered).
    /// Returns Ok(()) if all found within timeout, Err with missing signals otherwise.
    pub async fn wait_for_all_signals(
        &self,
        required: &[&[&str]],
        timeout_secs: f64,
    ) -> Result<(), Vec<String>> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f64(timeout_secs);

        loop {
            let log = self.serial_log.lock().await;
            let missing: Vec<String> = required
                .iter()
                .filter(|alts| !alts.iter().all(|sig| log.contains(sig)))
                .map(|alts| alts.join(" AND "))
                .collect();

            if missing.is_empty() {
                return Ok(());
            }

            drop(log);

            if start.elapsed() > timeout {
                return Err(missing);
            }

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    /// Check if any liveness signal is present.
    pub async fn has_liveness_signal(&self) -> bool {
        let log = self.serial_log.lock().await;
        LIVENESS_SIGNALS.iter().any(|sig| log.contains(sig))
    }

    /// Wait for any liveness signal within timeout.
    pub async fn wait_for_liveness(&self, timeout_secs: f64) -> bool {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs_f64(timeout_secs);

        loop {
            if self.has_liveness_signal().await {
                return true;
            }

            if start.elapsed() > timeout {
                return false;
            }

            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}
