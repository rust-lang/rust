use std::error::Error;
use std::process::Command;

fn output(cmd: &mut Command) -> Result<String, Box<dyn Error>> {
    Ok(String::from_utf8(cmd.output()?.stdout)?.trim().into())
}

fn default_build_triple() -> Result<String, Box<dyn Error>> {
    let ostype = output(Command::new("uname").arg("-s"));
    let cputype = output(Command::new("uname").arg("-m"));
    let (ostype, cputype) = match (ostype, cputype) {
        (Ok(ostype), Ok(cputype)) => (ostype, cputype),
        (ostype, cputype) => {
            if cfg!(windows) {
                return Ok("x86_64-pc-windows-msvc".into());
            } else {
                return ostype.and(cputype);
            }
        }
    };

    let mut cputype = cputype;
    let ostype = match ostype.as_str() {
        "Darwin" => "apple-darwin",
        "DragonFly" => "unknown-dragonfly",
        "FreeBSD" => "unknown-freebsd",
        "Haiku" => "unknown-haiku",
        "NetBSD" => "unknown-netbsd",
        "OpenBSD" => "unknown-openbsd",
        "Linux" => match output(Command::new("uname").arg("-o"))?.as_str() {
            "Android" => "linux-android",
            _ => "unknown-linux-gnu",
        },
        "SunOS" => {
            cputype = output(Command::new("isainfo").arg("-k"))?;
            "sun-solaris"
        }
        ostype if ostype.starts_with("MINGW") => {
            cputype = if std::env::var("MSYSTEM").map(|x| x == "MINGW64").unwrap_or_default() {
                "x86_64"
            } else {
                "i686"
            }
            .into();
            "pc-windows-gnu"
        }
        ostype if ostype.starts_with("MSYS") => "pc-windows-gnu",
        ostype if ostype.starts_with("CYGWIN_NT") => {
            cputype = if ostype.ends_with("WOW64") { "x86_64" } else { "i686" }.into();
            "pc-windows-gnu"
        }
        _ => {
            return Err(format!("unknown OS type: {}", ostype).into());
        }
    };

    if cputype == "powerpc" && ostype == "unknown-freebsd" {
        cputype = output(Command::new("uname").arg("-p"))?;
    }

    let mut ostype: String = ostype.into();
    let mut set_cputype_from_uname_p = false;
    match cputype.as_str() {
        "BePC" => "i686",
        "aarch64" => "aarch64",
        "amd64" => "x86_64",
        "arm64" => "aarch64",
        "i386" => "i686",
        "i486" => "i686",
        "i686" => "i686",
        "i786" => "i686",
        "powerpc" => "powerpc",
        "powerpc64" => "powerpc64",
        "powerpc64le" => "powerpc64le",
        "ppc" => "powerpc",
        "ppc64" => "powerpc64",
        "ppc64le" => "powerpc64le",
        "s390x" => "s390x",
        "x64" => "x86_64",
        "x86" => "i686",
        "x86-64" => "x86_64",
        "x86_64" => "x86_64",
        "xscale" | "arm" => match ostype.as_str() {
            "linux-android" => {
                ostype = "linux-androideabi".into();
                "arm"
            }
            "unknown-freebsd" => {
                ostype = "unknown-freebsd".into();
                set_cputype_from_uname_p = true;
                ""
            }
            _ => cputype.as_str(),
        },
        "armv6l" => {
            match ostype.as_str() {
                "linux-android" => {
                    ostype = "linux-androideabi".into();
                }
                _ => {
                    ostype += "eabihf";
                }
            };
            "arm"
        }
        "armv7l" | "armv8l" => {
            match ostype.as_str() {
                "linux-android" => {
                    ostype = "linux-androideabi".into();
                }
                _ => {
                    ostype += "eabihf";
                }
            };
            "armv7"
        }
        "mips" => {
            if cfg!(target_endian = "big") {
                "mips"
            } else {
                "mipsel"
            }
        }
        "mips64" => {
            ostype += "abi64";
            if cfg!(target_endian = "big") { "mips64" } else { "mips64el" }
        }
        "sparc" => "sparc",
        "sparcv9" => "sparcv9",
        "sparc64" => "sparc64",
        _ => {
            return Err(format!("unknown cpu type: {}", cputype).into());
        }
    };
    let cputype = if set_cputype_from_uname_p {
        output(Command::new("uname").arg("-p"))?
    } else {
        cputype.into()
    };

    Ok(format!("{}-{}", cputype, ostype))
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CFG_RELEASE");
    println!("cargo:rerun-if-env-changed=CFG_RELEASE_CHANNEL");
    println!("cargo:rerun-if-env-changed=CFG_VERSION");
    println!("cargo:rerun-if-env-changed=CFG_VER_DATE");
    println!("cargo:rerun-if-env-changed=CFG_VER_HASH");
    println!("cargo:rerun-if-env-changed=CFG_PREFIX");
    println!("cargo:rerun-if-env-changed=CFG_VIRTUAL_RUST_SOURCE_BASE_DIR");
    println!("cargo:rerun-if-env-changed=CFG_COMPILER_HOST_TRIPLE");
    println!("cargo:rerun-if-env-changed=CFG_LIBDIR_RELATIVE");

    if std::env::var_os("CFG_COMPILER_HOST_TRIPLE").is_none() {
        println!(
            "cargo:rustc-env=CFG_COMPILER_HOST_TRIPLE={}",
            default_build_triple().expect("Unable to determine build triple not found")
        )
    };
}
