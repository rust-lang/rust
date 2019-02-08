use std::io;
use std::process::Command;
use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};

use Arch::*;

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum Arch {
    Armv7,
    Armv7s,
    Arm64,
    I386,
    X86_64
}

impl Arch {
    pub fn to_string(self) -> &'static str {
        match self {
            Armv7 => "armv7",
            Armv7s => "armv7s",
            Arm64 => "arm64",
            I386 => "i386",
            X86_64 => "x86_64"
        }
    }
}

pub fn get_sdk_root(sdk_name: &str) -> Result<String, String> {
    let res = Command::new("xcrun")
                      .arg("--show-sdk-path")
                      .arg("-sdk")
                      .arg(sdk_name)
                      .output()
                      .and_then(|output| {
                          if output.status.success() {
                              Ok(String::from_utf8(output.stdout).unwrap())
                          } else {
                              let error = String::from_utf8(output.stderr);
                              let error = format!("process exit with error: {}",
                                                  error.unwrap());
                              Err(io::Error::new(io::ErrorKind::Other,
                                                 &error[..]))
                          }
                      });

    match res {
        Ok(output) => Ok(output.trim().to_string()),
        Err(e) => Err(format!("failed to get {} SDK path: {}", sdk_name, e))
    }
}

fn build_pre_link_args(arch: Arch) -> Result<LinkArgs, String> {
    let sdk_name = match arch {
        Armv7 | Armv7s | Arm64 => "iphoneos",
        I386 | X86_64 => "iphonesimulator"
    };

    let arch_name = arch.to_string();

    let sdk_root = get_sdk_root(sdk_name)?;

    let mut args = LinkArgs::new();
    args.insert(LinkerFlavor::Gcc,
                vec!["-arch".to_string(),
                     arch_name.to_string(),
                     "-isysroot".to_string(),
                     sdk_root.clone(),
                     "-Wl,-syslibroot".to_string(),
                     sdk_root]);

    Ok(args)
}

fn target_cpu(arch: Arch) -> String {
    match arch {
        Armv7 => "cortex-a8", // iOS7 is supported on iPhone 4 and higher
        Armv7s => "cortex-a9",
        Arm64 => "cyclone",
        I386 => "yonah",
        X86_64 => "core2",
    }.to_string()
}

pub fn opts(arch: Arch) -> Result<TargetOptions, String> {
    let pre_link_args = build_pre_link_args(arch)?;
    Ok(TargetOptions {
        cpu: target_cpu(arch),
        dynamic_linking: false,
        executables: true,
        pre_link_args,
        has_elf_tls: false,
        eliminate_frame_pointer: false,
        .. super::apple_base::opts()
    })
}
