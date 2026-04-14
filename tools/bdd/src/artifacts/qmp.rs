use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::Mutex;

/// Global QMP stream (for reporter access to screenshots).
/// Kept as a path, connections are opened on-demand to avoid QEMU deadlocks from buffer overflows.
pub(crate) static QMP_STREAM: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

/// Set the global QMP stream (called from world after init).
pub async fn set_qmp_stream(path: Option<PathBuf>) {
    if let Some(cache) = QMP_STREAM.get() {
        let mut guard = cache.lock().await;
        *guard = path;
    }
}

/// Execute a QMP command on a specific stream.
pub async fn execute_on_stream(
    stream: &mut UnixStream,
    command: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Helper to read a QMP line with a total timeout
    async fn read_line(
        stream: &mut UnixStream,
        deadline: tokio::time::Instant,
    ) -> std::io::Result<String> {
        let mut buf = [0u8; 1];
        let mut line = String::new();
        loop {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Read deadline exceeded",
                ));
            }
            let remaining = deadline - now;
            match tokio::time::timeout(remaining, stream.read(&mut buf)).await {
                Ok(Ok(n)) if n > 0 => {
                    let c = buf[0] as char;
                    line.push(c);
                    if c == '\n' {
                        break;
                    }
                }
                Ok(Ok(0)) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "EOF",
                    ));
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "Read timeout",
                    ));
                }
                Ok(Ok(_)) => unreachable!(),
            }
        }
        Ok(line)
    }

    if let Err(e) = stream.write_all(command.as_bytes()).await {
        return Err(format!("Failed to send QMP command: {}", e).into());
    }
    if let Err(e) = stream.write_all(b"\n").await {
        return Err(format!("Failed to send QMP newline: {}", e).into());
    }

    // Set a generous deadline (5 seconds)
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);

    // Safety break for events
    let mut event_count = 0;
    const MAX_EVENTS: usize = 100;

    loop {
        match read_line(stream, deadline).await {
            Ok(res) => {
                if res.contains(r#""event":"#) {
                    event_count += 1;
                    if event_count > MAX_EVENTS {
                        return Err("Too many QMP events without response".into());
                    }
                    continue;
                }
                return Ok(res);
            }
            Err(e) => return Err(format!("Failed to read QMP response: {}", e).into()),
        }
    }
}

pub async fn connect_qmp(
    socket_path: &std::path::Path,
) -> Result<UnixStream, Box<dyn std::error::Error + Send + Sync>> {
    let mut stream = UnixStream::connect(socket_path).await?;
    let mut buf = vec![0u8; 4096];
    let _ = stream.readable().await;
    let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await?;
    stream
        .write_all(b"{\"execute\": \"qmp_capabilities\"}\n")
        .await?;
    let _ = stream.readable().await;
    let _ = tokio::io::AsyncReadExt::read(&mut stream, &mut buf).await?;
    Ok(stream)
}

async fn qmp_execute(command: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let mutex = QMP_STREAM.get().ok_or("Artifacts system not initialized")?;
    let guard = mutex.lock().await;

    let path = match guard.as_ref() {
        Some(p) => p.clone(),
        None => return Err("No QMP connection active".into()),
    };
    drop(guard); // Free lock during I/O

    let mut stream = connect_qmp(&path).await?;
    execute_on_stream(&mut stream, command).await
}

/// Take a screenshot using the global QMP socket (for reporter).
pub async fn take_screenshot_global(
    output_path: &std::path::Path,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Get absolute path for QEMU
    let ppm_path = output_path.with_extension("ppm");
    let ppm_abs = std::fs::canonicalize(output_path.parent().ok_or("Invalid output path")?)?
        .join(ppm_path.file_name().ok_or("Invalid PPM path")?);

    // Retry a few times if "device not ready" or similar transient errors occur
    let mut success = false;
    for _ in 0..3 {
        let screendump_cmd = format!(
            r#"{{"execute": "screendump", "arguments": {{"filename": "{}"}}}}"#,
            ppm_abs.display()
        );

        match qmp_execute(&screendump_cmd).await {
            Ok(resp) => {
                if !resp.contains("error") {
                    success = true;
                    break;
                }
                eprintln!("[bdd-debug] QMP returned error: {}", resp);
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("No QMP connection active")
                    || msg.contains("Broken pipe")
                    || msg.contains("EOF")
                {
                    return Err(e); // Fatal connection loss
                }
                eprintln!("[bdd-debug] QMP execute failed: {}", msg);
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        }
    }

    if !success {
        return Err("Failed to capture screenshot after retries".into());
    }

    // Wait for file to appear - increased to 40 iterations (2s) for reliability
    for _ in 0..40 {
        if ppm_path.exists() {
            // Found it! Small extra sleep to ensure flushed?
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            break;
        }
        let _ = tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    if !ppm_path.exists() {
        return Err(format!("Screenshot file not created at {}", ppm_path.display()).into());
    }

    // Convert PPM to PNG
    let png_path = output_path.with_extension("png");
    let img = image::open(&ppm_path)?;
    img.save(&png_path)?;
    let _ = std::fs::remove_file(&ppm_path);

    Ok(png_path)
}

/// Dump CPU registers using the global QMP socket (for reporter).
pub async fn dump_registers_global(
    output_path: &std::path::Path,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let info_regs_cmd =
        r#"{"execute": "human-monitor-command", "arguments": {"command-line": "info registers"}}"#;

    let response_str = match qmp_execute(info_regs_cmd).await {
        Ok(s) => s,
        Err(e) => return Err(e),
    };

    // Parse JSON response to extract the actual output
    let content = if let Some(start) = response_str.find("\"return\": \"") {
        let remainder = &response_str[start + 11..];
        if let Some(end) = remainder.rfind("\"}") {
            remainder[..end]
                .replace("\\r\\n", "\n")
                .replace("\\n", "\n")
                .replace("\\\"", "\"")
        } else {
            response_str
        }
    } else {
        response_str
    };

    std::fs::write(output_path, content)?;

    Ok(output_path.to_path_buf())
}
