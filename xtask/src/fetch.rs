use anyhow::{Context, Result, ensure};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn fetch() -> Result<()> {
    let root = project_root();
    let assets = root.join("assets");

    if !assets.exists() {
        fs::create_dir_all(&assets)?;
    }

    let vendor = root.join("vendor");
    if !vendor.exists() {
        fs::create_dir_all(&vendor)?;
    }

    fetch_limine(&vendor)?;
    fetch_ovmf(&vendor)?;
    fetch_fonts(&assets)?;
    fetch_icons(&assets)?;
    fetch_cursors(&assets)?;
    fetch_unifont(&assets)?;

    #[cfg(feature = "svg-cursors")]
    fetch_future_cursors(&assets)?;
    fetch_pciids(&assets)?;

    Ok(())
}

fn fetch_limine(vendor: &Path) -> Result<()> {
    println!("==> Fetching Limine...");
    require_tool("git")?;

    let limine_dir = vendor.join("limine");
    if limine_dir.join(".git").exists() {
        println!("    Limine repo already exists, skipping clone.");
    } else {
        if limine_dir.exists() {
            fs::remove_dir_all(&limine_dir)?;
        }
        run_cmd(
            Command::new("git")
                .arg("clone")
                .arg("--branch=v10.x-binary")
                .arg("--depth=1")
                .arg("https://github.com/limine-bootloader/limine.git")
                .arg(&limine_dir),
        )?;
    }

    let required = [
        "limine-bios.sys",
        "limine-bios-cd.bin",
        "limine-uefi-cd.bin",
        "BOOTX64.EFI",
        "BOOTAA64.EFI",
        "BOOTRISCV64.EFI",
        "BOOTLOONGARCH64.EFI",
    ];

    let mut all_present = true;
    for f in required {
        let p = limine_dir.join(f);
        if !p.exists() {
            all_present = false;
            break;
        }
    }

    let limine_exe = limine_dir.join("limine");
    if all_present && limine_exe.exists() {
        println!("    Limine artifacts and tool already present, skipping build.");
    } else {
        println!("    Building limine CLI tool...");
        run_cmd(Command::new("make").arg("-C").arg(&limine_dir))?;
    }

    println!("    Limine ready.");
    Ok(())
}

fn fetch_ovmf(vendor: &Path) -> Result<()> {
    println!("==> Fetching OVMF...");
    require_tool("curl")?;
    require_tool("tar")?;

    let ovmf_dir = vendor.join("ovmf");
    fs::create_dir_all(&ovmf_dir)?;

    let release = std::env::var("THINGOS_OVMF_RELEASE")
        .unwrap_or_else(|_| "edk2-stable202508-r1".to_string());

    let mappings = [
        ("x64/code.fd", "ovmf-code-x86_64.fd"),
        ("x64/vars.fd", "ovmf-vars-x86_64.fd"),
        ("aarch64/code.fd", "ovmf-code-aarch64.fd"),
        ("aarch64/vars.fd", "ovmf-vars-aarch64.fd"),
        ("riscv64/code.fd", "ovmf-code-riscv64.fd"),
        ("riscv64/vars.fd", "ovmf-vars-riscv64.fd"),
        ("loongarch64/code.fd", "ovmf-code-loongarch64.fd"),
        ("loongarch64/vars.fd", "ovmf-vars-loongarch64.fd"),
    ];

    let mut missing = false;
    for (_, dest_name) in mappings {
        if !ovmf_dir.join(dest_name).exists() {
            missing = true;
            break;
        }
    }

    if !missing {
        println!("    OVMF artifacts already present, skipping.");
        return Ok(());
    }

    let archive_name = format!("{release}-bin.tar.xz");
    let archive_url = format!(
        "https://github.com/rust-osdev/ovmf-prebuilt/releases/download/{}/{}",
        release, archive_name
    );
    let archive_path = ovmf_dir.join(&archive_name);

    println!("    Using OVMF release: {release}");
    if !archive_path.exists() {
        println!("    Downloading {archive_name}...");
        download_file(&archive_url, &archive_path)
            .with_context(|| format!("Failed to download {archive_url}"))?;
    } else {
        println!("    Reusing cached {archive_name}");
    }

    let extract_dir = ovmf_dir.join(format!("extract-{release}"));
    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir)?;
    }
    fs::create_dir_all(&extract_dir)?;

    println!("    Extracting OVMF...");
    run_cmd(
        Command::new("tar")
            .arg("-xJf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&extract_dir),
    )
    .context("Failed to unpack OVMF archive")?;

    let base = extract_dir.join(format!("{release}-bin"));
    for (src_rel, dest_name) in mappings {
        let src = base.join(src_rel);
        if src.exists() {
            let dest = ovmf_dir.join(dest_name);
            println!(
                "    Installing {} -> {}",
                src_rel,
                dest.file_name().unwrap().to_string_lossy()
            );
            fs::copy(&src, &dest).with_context(|| format!("Failed to install {dest_name}"))?;
        } else {
            eprintln!("    [WARNING] Missing OVMF artifact: {src_rel}");
        }
    }

    fs::remove_dir_all(&extract_dir).context("Failed to clean temporary OVMF extraction dir")?;

    Ok(())
}

fn fetch_fonts(assets: &Path) -> Result<()> {
    println!("==> Fetching Fonts...");
    require_tool("curl")?;
    require_tool("unzip")?;
    require_tool("gunzip")?;

    let fonts_dir = assets.join("fonts");
    fs::create_dir_all(&fonts_dir)?;

    let hack_dest = fonts_dir.join("Hack-Regular.ttf");
    if !hack_dest.exists() {
        println!("    Fetching Hack-Regular.ttf...");
        println!("    Downloading Hack-v3.003-ttf.zip");
        let zip_path = fonts_dir.join("temp_hack.zip");
        download_file(
            "https://github.com/source-foundry/Hack/releases/download/v3.003/Hack-v3.003-ttf.zip",
            &zip_path,
        )?;

        run_cmd(
            Command::new("unzip")
                .arg("-o")
                .arg(&zip_path)
                .current_dir(&fonts_dir),
        )?;

        let candidates = [
            fonts_dir.join("ttf/Hack-Regular.ttf"),
            fonts_dir.join("Hack-Regular.ttf"),
        ];

        let mut found = false;
        for c in candidates {
            if c.exists() {
                if c != hack_dest {
                    fs::rename(&c, &hack_dest)?;
                }
                found = true;
                break;
            }
        }

        if !found {
            eprintln!("    [WARNING] Could not locate Hack-Regular.ttf after unzip.");
        }

        let _ = fs::remove_file(&zip_path);
        let _ = fs::remove_dir_all(fonts_dir.join("ttf"));
    }

    let noto_fonts = [
        (
            "NotoSans-Regular.ttf",
            "https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        ),
        (
            "NotoSerif-Regular.ttf",
            "https://github.com/notofonts/noto-fonts/raw/HEAD/hinted/ttf/NotoSerif/NotoSerif-Regular.ttf",
        ),
        (
            "NotoSansSymbol-Regular.ttf",
            "https://github.com/notofonts/noto-fonts/raw/HEAD/hinted/ttf/NotoSansSymbols/NotoSansSymbols-Regular.ttf",
        ),
        (
            "NotoSansSymbol2-Regular.ttf",
            "https://github.com/notofonts/noto-fonts/raw/HEAD/hinted/ttf/NotoSansSymbols2/NotoSansSymbols2-Regular.ttf",
        ),
    ];

    for (name, url) in noto_fonts {
        let dest = fonts_dir.join(name);
        if !dest.exists() {
            println!("    Downloading {name}...");
            if let Err(e) = download_file(url, &dest) {
                eprintln!(
                    "    [WARNING] Failed to download {}: {}. Creating placeholder.",
                    name, e
                );
                fs::write(&dest, b"PLACEHOLDER FONT")?;
            }
        }
    }

    let dseg_dest = fonts_dir.join("DSEG7Classic-Regular.ttf");
    if !dseg_dest.exists() {
        println!("    Fetching DSEG7Classic-Regular.ttf...");
        let zip_path = fonts_dir.join("temp_dseg.zip");
        download_file(
            "https://github.com/keshikan/DSEG/releases/download/v0.46/fonts-DSEG_v046.zip",
            &zip_path,
        )?;

        run_cmd(
            Command::new("unzip")
                .arg("-o")
                .arg("-j")
                .arg(&zip_path)
                .arg("fonts-DSEG_v046/DSEG7-Classic/DSEG7Classic-Regular.ttf")
                .arg("-d")
                .arg(&fonts_dir),
        )?;

        let _ = fs::remove_file(&zip_path);
    }

    Ok(())
}

fn fetch_icons(assets: &Path) -> Result<()> {
    println!("==> Fetching Icons (Tango)...");
    require_tool("curl")?;
    require_tool("tar")?;

    let icons_dir = assets.join("icons");
    let tango_dir = icons_dir.join("tango");
    fs::create_dir_all(&icons_dir)?;

    if tango_dir.join("32x32").exists() {
        println!("    Tango icon set already exists.");
        return Ok(());
    }

    let root = project_root();
    let temp_dir = root.join("target/temp_icons");
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir)?;
    }
    fs::create_dir_all(&temp_dir)?;

    let tango_tar = temp_dir.join("tango.tar.gz");
    println!("    Downloading Tango icon set...");
    download_file(
        "http://tango.freedesktop.org/releases/tango-icon-theme-0.8.90.tar.gz",
        &tango_tar,
    )?;

    println!("    Extracting full Tango icon set...");
    run_cmd(
        Command::new("tar")
            .arg("-xzf")
            .arg(&tango_tar)
            .current_dir(&temp_dir),
    )?;

    let extracted = temp_dir.join("tango-icon-theme-0.8.90");
    if extracted.exists() {
        if tango_dir.exists() {
            fs::remove_dir_all(&tango_dir)?;
        }
        fs::rename(&extracted, &tango_dir)?;
        println!("    Installed Tango icon set to icons/tango/");
    }

    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}

fn fetch_cursors(assets: &Path) -> Result<()> {
    println!("==> Fetching Cursors...");
    let cursors_dir = assets.join("cursors");
    fs::create_dir_all(&cursors_dir)?;

    let plain_dir = cursors_dir.join("plain");
    if !plain_dir.exists() {
        println!("    Fetching Plain Cursors (.cur/.ani)...");
        require_tool("curl")?;
        require_tool("unzip")?;

        let url = "https://www.rw-designer.com/cursor-downloadset/plain.zip";
        let zip_path = cursors_dir.join("plain.zip");

        if download_file(url, &zip_path).is_ok() {
            println!("    Extracting Plain Cursors...");
            fs::create_dir_all(&plain_dir)?;
            run_cmd(
                Command::new("unzip")
                    .arg("-o")
                    .arg(&zip_path)
                    .arg("-d")
                    .arg(&plain_dir),
            )?;
            let _ = fs::remove_file(&zip_path);
        } else {
            eprintln!("    [WARNING] Failed to download Plain Cursors.");
        }
    } else {
        println!("    Plain Cursors already exist.");
    }

    Ok(())
}

fn fetch_pciids(assets: &Path) -> Result<()> {
    println!("==> Fetching pci.ids...");
    let pci_dir = assets.join("pci");
    fs::create_dir_all(&pci_dir)?;

    let pci_ids = pci_dir.join("pci.ids");
    if !pci_ids.exists() {
        println!("    Downloading pci.ids...");
        download_file("https://pci-ids.ucw.cz/v2.2/pci.ids", &pci_ids)
            .context("Failed to download pci.ids")?;
    } else {
        println!("    pci.ids already exists.");
    }

    Ok(())
}

fn fetch_unifont(assets: &Path) -> Result<()> {
    println!("==> Fetching Unifont...");
    require_tool("curl")?;
    require_tool("gunzip")?;

    let fonts_dir = assets.join("fonts");
    fs::create_dir_all(&fonts_dir)?;

    let unifont_dest = fonts_dir.join("unifont.hex");
    if !unifont_dest.exists() {
        println!("    Downloading unifont_all.hex.gz...");
        let gz_path = fonts_dir.join("unifont_all.hex.gz");
        download_file(
            "https://unifoundry.com/pub/unifont/unifont-17.0.04/font-builds/unifont_all-17.0.04.hex.gz",
            &gz_path,
        )?;

        println!("    Extracting unifont.hex...");
        run_cmd(Command::new("gunzip").arg("-f").arg(&gz_path))?;

        let extracted = fonts_dir.join("unifont_all.hex");
        if extracted.exists() {
            fs::rename(&extracted, &unifont_dest)?;
        } else {
            let versioned = fonts_dir.join("unifont_all-17.0.04.hex");
            if versioned.exists() {
                fs::rename(&versioned, &unifont_dest)?;
            }
        }
    } else {
        println!("    unifont.hex already exists.");
    }

    Ok(())
}

fn require_tool(tool: &str) -> Result<()> {
    if Command::new("which").arg(tool).output().is_err() {
        anyhow::bail!("Missing required tool: {tool}");
    }
    Ok(())
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    run_cmd(
        Command::new("curl")
            .arg("-f")
            .arg("-L")
            .arg("-o")
            .arg(dest)
            .arg(url),
    )
}

fn run_cmd(cmd: &mut Command) -> Result<()> {
    let status = cmd
        .status()
        .with_context(|| format!("Failed to run {:?}", cmd))?;
    ensure!(status.success(), "Command failed: {:?}", cmd);
    Ok(())
}

fn project_root() -> PathBuf {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    Path::new(&manifest_dir).parent().unwrap().to_path_buf()
}

#[cfg(feature = "svg-cursors")]
fn fetch_future_cursors(assets: &Path) -> Result<()> {
    println!("==> Fetching Future Cursors (SVG)...");
    require_tool("git")?;

    let root = project_root();
    let vendor = root.join("vendor");
    let future_dir = vendor.join("future-cursors");

    if !future_dir.join(".git").exists() {
        if future_dir.exists() {
            fs::remove_dir_all(&future_dir)?;
        }
        println!("    Cloning Future-cursors repo...");
        run_cmd(
            Command::new("git")
                .arg("clone")
                .arg("--depth=1")
                .arg("https://github.com/yeyushengfan258/Future-cursors")
                .arg(&future_dir),
        )?;
    } else {
        println!("    Future-cursors repo already exists.");
    }

    let dest_dir = assets.join("cursors/future");
    if dest_dir.exists() {
        fs::remove_dir_all(&dest_dir)?;
    }
    fs::create_dir_all(&dest_dir)?;

    let src_dir = future_dir.join("src/svg");
    if src_dir.exists() {
        for entry in fs::read_dir(&src_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("svg") {
                let file_name = path.file_name().unwrap();
                fs::copy(&path, dest_dir.join(file_name))?;
            }
        }
        println!("    Future SVG cursors copied to assets.");
    } else {
        eprintln!("    [WARNING] src/svg not found in Future-cursors repo.");
    }

    Ok(())
}
