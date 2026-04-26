#![feature(doc_cfg)]

//@ has all_targets/fn.foo.html \
//  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on GNU or Catalyst or Managarm C Library or MSVC or musl or Newlib or \
//  Neutrino 7.0 or Neutrino 7.1 or Neutrino 7.1 with io-sock or Neutrino 8.0 or \
//  OpenHarmony or relibc or SGX or Simulator or WASIp1 or WASIp2 or WASIp3 or \
//  uClibc or V5 or target_env=fake_env only.'
#[doc(cfg(any(
    target_env = "gnu",
    target_env = "macabi",
    target_env = "mlibc",
    target_env = "msvc",
    target_env = "musl",
    target_env = "newlib",
    target_env = "nto70",
    target_env = "nto71",
    target_env = "nto71_iosock",
    target_env = "nto80",
    target_env = "ohos",
    target_env = "relibc",
    target_env = "sgx",
    target_env = "sim",
    target_env = "p1",
    target_env = "p2",
    target_env = "p3",
    target_env = "uclibc",
    target_env = "v5",
    target_env = "fake_env",
)))]
pub fn foo() {}

//@ has all_targets/fn.bar.html \
//  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on AArch64 or AMD GPU or ARM or ARM64EC or AVR or BPF or C-SKY or \
//  Hexagon or LoongArch32 or LoongArch64 or Motorola 680x0 or MIPS or MIPS release \
//  6 or MIPS-64 or MIPS-64 release 6 or MSP430 or NVidia GPU or PowerPC or \
//  PowerPC64 or RISC-V RV32 or RISC-V RV64 or s390x or SPARC or SPARC-64 or SPIR-V \
//  or WebAssembly or WebAssembly or x86 or x86-64 or Xtensa or \
//  target_arch=fake_arch only.'
#[doc(cfg(any(
    target_arch = "aarch64",
    target_arch = "amdgpu",
    target_arch = "arm",
    target_arch = "arm64ec",
    target_arch = "avr",
    target_arch = "bpf",
    target_arch = "csky",
    target_arch = "hexagon",
    target_arch = "loongarch32",
    target_arch = "loongarch64",
    target_arch = "m68k",
    target_arch = "mips",
    target_arch = "mips32r6",
    target_arch = "mips64",
    target_arch = "mips64r6",
    target_arch = "msp430",
    target_arch = "nvptx64",
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "riscv32",
    target_arch = "riscv64",
    target_arch = "s390x",
    target_arch = "sparc",
    target_arch = "sparc64",
    target_arch = "spirv",
    target_arch = "wasm32",
    target_arch = "wasm64",
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "xtensa",
    target_arch = "fake_arch",
)))]
pub fn bar() {}

//@ has all_targets/fn.baz.html \
//  '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//  'Available on AIX and AMD HSA and Android and CUDA and Cygwin and DragonFly \
//  BSD and Emscripten and ESP-IDF and FreeBSD and Fuchsia and Haiku and HelenOS \
//  and Hermit and Horizon and GNU/Hurd and illumos and iOS and L4Re and Linux \
//  and LynxOS-178 and macOS and Managarm and Motor OS and NetBSD and bare-metal \
//  and QNX Neutrino and NuttX and OpenBSD and Play Station Portable and Play \
//  Station 1 and QuRT and Redox OS and RTEMS OS and Solaris and SOLID ASP3 and \
//  TEEOS and Trusty and tvOS and UEFI and VEXos and visionOS and Play Station \
//  Vita and VxWorks and WASI and watchOS and Windows and Xous and zero knowledge \
//  Virtual Machine and target_os=unknown and target_os=fake_os only.'
#[doc(cfg(all(
    target_os = "aix",
    target_os = "amdhsa",
    target_os = "android",
    target_os = "cuda",
    target_os = "cygwin",
    target_os = "dragonfly",
    target_os = "emscripten",
    target_os = "espidf",
    target_os = "freebsd",
    target_os = "fuchsia",
    target_os = "haiku",
    target_os = "helenos",
    target_os = "hermit",
    target_os = "horizon",
    target_os = "hurd",
    target_os = "illumos",
    target_os = "ios",
    target_os = "l4re",
    target_os = "linux",
    target_os = "lynxos178",
    target_os = "macos",
    target_os = "managarm",
    target_os = "motor",
    target_os = "netbsd",
    target_os = "none",
    target_os = "nto",
    target_os = "nuttx",
    target_os = "openbsd",
    target_os = "psp",
    target_os = "psx",
    target_os = "qurt",
    target_os = "redox",
    target_os = "rtems",
    target_os = "solaris",
    target_os = "solid_asp3",
    target_os = "teeos",
    target_os = "trusty",
    target_os = "tvos",
    target_os = "uefi",
    target_os = "vexos",
    target_os = "visionos",
    target_os = "vita",
    target_os = "vxworks",
    target_os = "wasi",
    target_os = "watchos",
    target_os = "windows",
    target_os = "xous",
    target_os = "zkvm",
    target_os = "unknown",
    target_os = "fake_os",
)))]
pub fn baz() {}
