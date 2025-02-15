use rustc_target::spec::Target;
pub(super) use rustc_target::spec::apple::OSVersion;

#[cfg(test)]
mod tests;

pub(super) fn macho_platform(target: &Target) -> u32 {
    match (&*target.os, &*target.abi) {
        ("macos", _) => object::macho::PLATFORM_MACOS,
        ("ios", "macabi") => object::macho::PLATFORM_MACCATALYST,
        ("ios", "sim") => object::macho::PLATFORM_IOSSIMULATOR,
        ("ios", _) => object::macho::PLATFORM_IOS,
        ("watchos", "sim") => object::macho::PLATFORM_WATCHOSSIMULATOR,
        ("watchos", _) => object::macho::PLATFORM_WATCHOS,
        ("tvos", "sim") => object::macho::PLATFORM_TVOSSIMULATOR,
        ("tvos", _) => object::macho::PLATFORM_TVOS,
        ("visionos", "sim") => object::macho::PLATFORM_XROSSIMULATOR,
        ("visionos", _) => object::macho::PLATFORM_XROS,
        _ => unreachable!("tried to get Mach-O platform for non-Apple target"),
    }
}

pub(super) fn add_version_to_llvm_target(
    llvm_target: &str,
    deployment_target: OSVersion,
) -> String {
    let mut components = llvm_target.split("-");
    let arch = components.next().expect("apple target should have arch");
    let vendor = components.next().expect("apple target should have vendor");
    let os = components.next().expect("apple target should have os");
    let environment = components.next();
    assert_eq!(components.next(), None, "too many LLVM triple components");

    assert!(
        !os.contains(|c: char| c.is_ascii_digit()),
        "LLVM target must not already be versioned"
    );

    let version = deployment_target.fmt_full();
    if let Some(env) = environment {
        // Insert version into OS, before environment
        format!("{arch}-{vendor}-{os}{version}-{env}")
    } else {
        format!("{arch}-{vendor}-{os}{version}")
    }
}
