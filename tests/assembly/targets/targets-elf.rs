//@ add-core-stubs
//@ assembly-output: emit-asm
// ignore-tidy-linelength
//@ revisions: aarch64_be_unknown_linux_gnu
//@ [aarch64_be_unknown_linux_gnu] compile-flags: --target aarch64_be-unknown-linux-gnu
//@ [aarch64_be_unknown_linux_gnu] needs-llvm-components: aarch64
//@ revisions: aarch64_be_unknown_linux_gnu_ilp32
//@ [aarch64_be_unknown_linux_gnu_ilp32] compile-flags: --target aarch64_be-unknown-linux-gnu_ilp32
//@ [aarch64_be_unknown_linux_gnu_ilp32] needs-llvm-components: aarch64
//@ revisions: aarch64_be_unknown_netbsd
//@ [aarch64_be_unknown_netbsd] compile-flags: --target aarch64_be-unknown-netbsd
//@ [aarch64_be_unknown_netbsd] needs-llvm-components: aarch64
//@ revisions: aarch64_kmc_solid_asp3
//@ [aarch64_kmc_solid_asp3] compile-flags: --target aarch64-kmc-solid_asp3
//@ [aarch64_kmc_solid_asp3] needs-llvm-components: aarch64
//@ revisions: aarch64_linux_android
//@ [aarch64_linux_android] compile-flags: --target aarch64-linux-android
//@ [aarch64_linux_android] needs-llvm-components: aarch64
//@ revisions: aarch64_nintendo_switch_freestanding
//@ [aarch64_nintendo_switch_freestanding] compile-flags: --target aarch64-nintendo-switch-freestanding
//@ [aarch64_nintendo_switch_freestanding] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_freebsd
//@ [aarch64_unknown_freebsd] compile-flags: --target aarch64-unknown-freebsd
//@ [aarch64_unknown_freebsd] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_fuchsia
//@ [aarch64_unknown_fuchsia] compile-flags: --target aarch64-unknown-fuchsia
//@ [aarch64_unknown_fuchsia] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_hermit
//@ [aarch64_unknown_hermit] compile-flags: --target aarch64-unknown-hermit
//@ [aarch64_unknown_hermit] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_illumos
//@ [aarch64_unknown_illumos] compile-flags: --target aarch64-unknown-illumos
//@ [aarch64_unknown_illumos] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_linux_gnu
//@ [aarch64_unknown_linux_gnu] compile-flags: --target aarch64-unknown-linux-gnu
//@ [aarch64_unknown_linux_gnu] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_linux_gnu_ilp32
//@ [aarch64_unknown_linux_gnu_ilp32] compile-flags: --target aarch64-unknown-linux-gnu_ilp32
//@ [aarch64_unknown_linux_gnu_ilp32] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_linux_musl
//@ [aarch64_unknown_linux_musl] compile-flags: --target aarch64-unknown-linux-musl
//@ [aarch64_unknown_linux_musl] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_linux_ohos
//@ [aarch64_unknown_linux_ohos] compile-flags: --target aarch64-unknown-linux-ohos
//@ [aarch64_unknown_linux_ohos] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_netbsd
//@ [aarch64_unknown_netbsd] compile-flags: --target aarch64-unknown-netbsd
//@ [aarch64_unknown_netbsd] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_none
//@ [aarch64_unknown_none] compile-flags: --target aarch64-unknown-none
//@ [aarch64_unknown_none] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_none_softfloat
//@ [aarch64_unknown_none_softfloat] compile-flags: --target aarch64-unknown-none-softfloat
//@ [aarch64_unknown_none_softfloat] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_nto_qnx700
//@ [aarch64_unknown_nto_qnx700] compile-flags: --target aarch64-unknown-nto-qnx700
//@ [aarch64_unknown_nto_qnx700] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_nto_qnx710
//@ [aarch64_unknown_nto_qnx710] compile-flags: --target aarch64-unknown-nto-qnx710
//@ [aarch64_unknown_nto_qnx710] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_nto_qnx710_iosock
//@ [aarch64_unknown_nto_qnx710_iosock] compile-flags: --target aarch64-unknown-nto-qnx710_iosock
//@ [aarch64_unknown_nto_qnx710_iosock] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_nto_qnx800
//@ [aarch64_unknown_nto_qnx800] compile-flags: --target aarch64-unknown-nto-qnx800
//@ [aarch64_unknown_nto_qnx800] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_openbsd
//@ [aarch64_unknown_openbsd] compile-flags: --target aarch64-unknown-openbsd
//@ [aarch64_unknown_openbsd] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_redox
//@ [aarch64_unknown_redox] compile-flags: --target aarch64-unknown-redox
//@ [aarch64_unknown_redox] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_teeos
//@ [aarch64_unknown_teeos] compile-flags: --target aarch64-unknown-teeos
//@ [aarch64_unknown_teeos] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_nuttx
//@ [aarch64_unknown_nuttx] compile-flags: --target aarch64-unknown-nuttx
//@ [aarch64_unknown_nuttx] needs-llvm-components: aarch64
//@ revisions: aarch64_unknown_trusty
//@ [aarch64_unknown_trusty] compile-flags: --target aarch64-unknown-trusty
//@ [aarch64_unknown_trusty] needs-llvm-components: aarch64
//@ revisions: aarch64_wrs_vxworks
//@ [aarch64_wrs_vxworks] compile-flags: --target aarch64-wrs-vxworks
//@ [aarch64_wrs_vxworks] needs-llvm-components: aarch64
//@ revisions: arm_linux_androideabi
//@ [arm_linux_androideabi] compile-flags: --target arm-linux-androideabi
//@ [arm_linux_androideabi] needs-llvm-components: arm
//@ revisions: arm_unknown_linux_gnueabi
//@ [arm_unknown_linux_gnueabi] compile-flags: --target arm-unknown-linux-gnueabi
//@ [arm_unknown_linux_gnueabi] needs-llvm-components: arm
//@ revisions: arm_unknown_linux_gnueabihf
//@ [arm_unknown_linux_gnueabihf] compile-flags: --target arm-unknown-linux-gnueabihf
//@ [arm_unknown_linux_gnueabihf] needs-llvm-components: arm
//@ revisions: arm_unknown_linux_musleabi
//@ [arm_unknown_linux_musleabi] compile-flags: --target arm-unknown-linux-musleabi
//@ [arm_unknown_linux_musleabi] needs-llvm-components: arm
//@ revisions: arm_unknown_linux_musleabihf
//@ [arm_unknown_linux_musleabihf] compile-flags: --target arm-unknown-linux-musleabihf
//@ [arm_unknown_linux_musleabihf] needs-llvm-components: arm
//@ revisions: armeb_unknown_linux_gnueabi
//@ [armeb_unknown_linux_gnueabi] compile-flags: --target armeb-unknown-linux-gnueabi
//@ [armeb_unknown_linux_gnueabi] needs-llvm-components: arm
//@ revisions: armebv7r_none_eabi
//@ [armebv7r_none_eabi] compile-flags: --target armebv7r-none-eabi
//@ [armebv7r_none_eabi] needs-llvm-components: arm
//@ revisions: armebv7r_none_eabihf
//@ [armebv7r_none_eabihf] compile-flags: --target armebv7r-none-eabihf
//@ [armebv7r_none_eabihf] needs-llvm-components: arm
//@ revisions: armv4t_none_eabi
//@ [armv4t_none_eabi] compile-flags: --target armv4t-none-eabi
//@ [armv4t_none_eabi] needs-llvm-components: arm
//@ revisions: armv4t_unknown_linux_gnueabi
//@ [armv4t_unknown_linux_gnueabi] compile-flags: --target armv4t-unknown-linux-gnueabi
//@ [armv4t_unknown_linux_gnueabi] needs-llvm-components: arm
//@ revisions: armv5te_none_eabi
//@ [armv5te_none_eabi] compile-flags: --target armv5te-none-eabi
//@ [armv5te_none_eabi] needs-llvm-components: arm
//@ revisions: armv5te_unknown_linux_gnueabi
//@ [armv5te_unknown_linux_gnueabi] compile-flags: --target armv5te-unknown-linux-gnueabi
//@ [armv5te_unknown_linux_gnueabi] needs-llvm-components: arm
//@ revisions: armv5te_unknown_linux_musleabi
//@ [armv5te_unknown_linux_musleabi] compile-flags: --target armv5te-unknown-linux-musleabi
//@ [armv5te_unknown_linux_musleabi] needs-llvm-components: arm
//@ revisions: armv5te_unknown_linux_uclibceabi
//@ [armv5te_unknown_linux_uclibceabi] compile-flags: --target armv5te-unknown-linux-uclibceabi
//@ [armv5te_unknown_linux_uclibceabi] needs-llvm-components: arm
//@ revisions: armv6_unknown_freebsd
//@ [armv6_unknown_freebsd] compile-flags: --target armv6-unknown-freebsd
//@ [armv6_unknown_freebsd] needs-llvm-components: arm
//@ revisions: armv6_unknown_netbsd_eabihf
//@ [armv6_unknown_netbsd_eabihf] compile-flags: --target armv6-unknown-netbsd-eabihf
//@ [armv6_unknown_netbsd_eabihf] needs-llvm-components: arm
//@ revisions: armv6k_nintendo_3ds
//@ [armv6k_nintendo_3ds] compile-flags: --target armv6k-nintendo-3ds
//@ [armv6k_nintendo_3ds] needs-llvm-components: arm
//@ revisions: armv7_linux_androideabi
//@ [armv7_linux_androideabi] compile-flags: --target armv7-linux-androideabi
//@ [armv7_linux_androideabi] needs-llvm-components: arm
//@ revisions: armv7_rtems_eabihf
//@ [armv7_rtems_eabihf] compile-flags: --target armv7-rtems-eabihf
//@ [armv7_rtems_eabihf] needs-llvm-components: arm
//@ revisions: armv7_sony_vita_newlibeabihf
//@ [armv7_sony_vita_newlibeabihf] compile-flags: --target armv7-sony-vita-newlibeabihf
//@ [armv7_sony_vita_newlibeabihf] needs-llvm-components: arm
//@ revisions: armv7_unknown_freebsd
//@ [armv7_unknown_freebsd] compile-flags: --target armv7-unknown-freebsd
//@ [armv7_unknown_freebsd] needs-llvm-components: arm
//@ revisions: armv7_unknown_linux_gnueabi
//@ [armv7_unknown_linux_gnueabi] compile-flags: --target armv7-unknown-linux-gnueabi
//@ [armv7_unknown_linux_gnueabi] needs-llvm-components: arm
//@ revisions: armv7_unknown_linux_gnueabihf
//@ [armv7_unknown_linux_gnueabihf] compile-flags: --target armv7-unknown-linux-gnueabihf
//@ [armv7_unknown_linux_gnueabihf] needs-llvm-components: arm
//@ revisions: armv7_unknown_linux_musleabi
//@ [armv7_unknown_linux_musleabi] compile-flags: --target armv7-unknown-linux-musleabi
//@ [armv7_unknown_linux_musleabi] needs-llvm-components: arm
//@ revisions: armv7_unknown_linux_musleabihf
//@ [armv7_unknown_linux_musleabihf] compile-flags: --target armv7-unknown-linux-musleabihf
//@ [armv7_unknown_linux_musleabihf] needs-llvm-components: arm
//@ revisions: armv7_unknown_linux_ohos
//@ [armv7_unknown_linux_ohos] compile-flags: --target armv7-unknown-linux-ohos
//@ [armv7_unknown_linux_ohos] needs-llvm-components: arm
//@ revisions: armv7_unknown_linux_uclibceabi
//@ [armv7_unknown_linux_uclibceabi] compile-flags: --target armv7-unknown-linux-uclibceabi
//@ [armv7_unknown_linux_uclibceabi] needs-llvm-components: arm
//@ revisions: armv7_unknown_linux_uclibceabihf
//@ [armv7_unknown_linux_uclibceabihf] compile-flags: --target armv7-unknown-linux-uclibceabihf
//@ [armv7_unknown_linux_uclibceabihf] needs-llvm-components: arm
//@ revisions: armv7_unknown_netbsd_eabihf
//@ [armv7_unknown_netbsd_eabihf] compile-flags: --target armv7-unknown-netbsd-eabihf
//@ [armv7_unknown_netbsd_eabihf] needs-llvm-components: arm
//@ revisions: armv7_unknown_trusty
//@ [armv7_unknown_trusty] compile-flags: --target armv7-unknown-trusty
//@ [armv7_unknown_trusty] needs-llvm-components: arm
//@ revisions: armv7_wrs_vxworks_eabihf
//@ [armv7_wrs_vxworks_eabihf] compile-flags: --target armv7-wrs-vxworks-eabihf
//@ [armv7_wrs_vxworks_eabihf] needs-llvm-components: arm
//@ revisions: armv7a_kmc_solid_asp3_eabi
//@ [armv7a_kmc_solid_asp3_eabi] compile-flags: --target armv7a-kmc-solid_asp3-eabi
//@ [armv7a_kmc_solid_asp3_eabi] needs-llvm-components: arm
//@ revisions: armv7a_kmc_solid_asp3_eabihf
//@ [armv7a_kmc_solid_asp3_eabihf] compile-flags: --target armv7a-kmc-solid_asp3-eabihf
//@ [armv7a_kmc_solid_asp3_eabihf] needs-llvm-components: arm
//@ revisions: armv7a_none_eabi
//@ [armv7a_none_eabi] compile-flags: --target armv7a-none-eabi
//@ [armv7a_none_eabi] needs-llvm-components: arm
//@ revisions: armv7a_none_eabihf
//@ [armv7a_none_eabihf] compile-flags: --target armv7a-none-eabihf
//@ [armv7a_none_eabihf] needs-llvm-components: arm
//@ revisions: armv7a_nuttx_eabi
//@ [armv7a_nuttx_eabi] compile-flags: --target armv7a-nuttx-eabi
//@ [armv7a_nuttx_eabi] needs-llvm-components: arm
//@ revisions: armv7a_nuttx_eabihf
//@ [armv7a_nuttx_eabihf] compile-flags: --target armv7a-nuttx-eabihf
//@ [armv7a_nuttx_eabihf] needs-llvm-components: arm
//@ revisions: armv7r_none_eabi
//@ [armv7r_none_eabi] compile-flags: --target armv7r-none-eabi
//@ [armv7r_none_eabi] needs-llvm-components: arm
//@ revisions: armv7r_none_eabihf
//@ [armv7r_none_eabihf] compile-flags: --target armv7r-none-eabihf
//@ [armv7r_none_eabihf] needs-llvm-components: arm
//@ revisions: armv8r_none_eabihf
//@ [armv8r_none_eabihf] compile-flags: --target armv8r-none-eabihf
//@ [armv8r_none_eabihf] needs-llvm-components: arm
// FIXME: disabled since it fails on CI saying the csky component is missing
/*
    revisions: csky_unknown_linux_gnuabiv2
    [csky_unknown_linux_gnuabiv2] compile-flags: --target csky-unknown-linux-gnuabiv2
    [csky_unknown_linux_gnuabiv2] needs-llvm-components: csky
    revisions: csky_unknown_linux_gnuabiv2hf
    [csky_unknown_linux_gnuabiv2hf] compile-flags: --target csky-unknown-linux-gnuabiv2hf
    [csky_unknown_linux_gnuabiv2hf] needs-llvm-components: csky
*/
//@ revisions: hexagon_unknown_linux_musl
//@ [hexagon_unknown_linux_musl] compile-flags: --target hexagon-unknown-linux-musl
//@ [hexagon_unknown_linux_musl] needs-llvm-components: hexagon
//@ revisions: hexagon_unknown_none_elf
//@ [hexagon_unknown_none_elf] compile-flags: --target hexagon-unknown-none-elf
//@ [hexagon_unknown_none_elf] needs-llvm-components: hexagon
//@ revisions: i686_pc_nto_qnx700
//@ [i686_pc_nto_qnx700] compile-flags: --target i686-pc-nto-qnx700
//@ [i686_pc_nto_qnx700] needs-llvm-components: x86
//@ revisions: i586_unknown_linux_gnu
//@ [i586_unknown_linux_gnu] compile-flags: --target i586-unknown-linux-gnu
//@ [i586_unknown_linux_gnu] needs-llvm-components: x86
//@ revisions: i586_unknown_linux_musl
//@ [i586_unknown_linux_musl] compile-flags: --target i586-unknown-linux-musl
//@ [i586_unknown_linux_musl] needs-llvm-components: x86
//@ revisions: i586_unknown_netbsd
//@ [i586_unknown_netbsd] compile-flags: --target i586-unknown-netbsd
//@ [i586_unknown_netbsd] needs-llvm-components: x86
//@ revisions: i586_unknown_redox
//@ [i586_unknown_redox] compile-flags: --target i586-unknown-redox
//@ [i586_unknown_redox] needs-llvm-components: x86
//@ revisions: i686_linux_android
//@ [i686_linux_android] compile-flags: --target i686-linux-android
//@ [i686_linux_android] needs-llvm-components: x86
//@ revisions: i686_unknown_freebsd
//@ [i686_unknown_freebsd] compile-flags: --target i686-unknown-freebsd
//@ [i686_unknown_freebsd] needs-llvm-components: x86
//@ revisions: i686_unknown_haiku
//@ [i686_unknown_haiku] compile-flags: --target i686-unknown-haiku
//@ [i686_unknown_haiku] needs-llvm-components: x86
//@ revisions: i686_unknown_hurd_gnu
//@ [i686_unknown_hurd_gnu] compile-flags: --target i686-unknown-hurd-gnu
//@ [i686_unknown_hurd_gnu] needs-llvm-components: x86
//@ revisions: i686_unknown_linux_gnu
//@ [i686_unknown_linux_gnu] compile-flags: --target i686-unknown-linux-gnu
//@ [i686_unknown_linux_gnu] needs-llvm-components: x86
//@ revisions: i686_unknown_linux_musl
//@ [i686_unknown_linux_musl] compile-flags: --target i686-unknown-linux-musl
//@ [i686_unknown_linux_musl] needs-llvm-components: x86
//@ revisions: i686_unknown_netbsd
//@ [i686_unknown_netbsd] compile-flags: --target i686-unknown-netbsd
//@ [i686_unknown_netbsd] needs-llvm-components: x86
//@ revisions: i686_unknown_openbsd
//@ [i686_unknown_openbsd] compile-flags: --target i686-unknown-openbsd
//@ [i686_unknown_openbsd] needs-llvm-components: x86
//@ revisions: i686_wrs_vxworks
//@ [i686_wrs_vxworks] compile-flags: --target i686-wrs-vxworks
//@ [i686_wrs_vxworks] needs-llvm-components: x86
//@ revisions: loongarch32_unknown_none
//@ [loongarch32_unknown_none] compile-flags: --target loongarch32-unknown-none
//@ [loongarch32_unknown_none] needs-llvm-components: loongarch
//@ revisions: loongarch32_unknown_none_softfloat
//@ [loongarch32_unknown_none_softfloat] compile-flags: --target loongarch32-unknown-none-softfloat
//@ [loongarch32_unknown_none_softfloat] needs-llvm-components: loongarch
//@ revisions: loongarch64_unknown_linux_gnu
//@ [loongarch64_unknown_linux_gnu] compile-flags: --target loongarch64-unknown-linux-gnu
//@ [loongarch64_unknown_linux_gnu] needs-llvm-components: loongarch
//@ revisions: loongarch64_unknown_linux_musl
//@ [loongarch64_unknown_linux_musl] compile-flags: --target loongarch64-unknown-linux-musl
//@ [loongarch64_unknown_linux_musl] needs-llvm-components: loongarch
//@ revisions: loongarch64_unknown_linux_ohos
//@ [loongarch64_unknown_linux_ohos] compile-flags: --target loongarch64-unknown-linux-ohos
//@ [loongarch64_unknown_linux_ohos] needs-llvm-components: loongarch
//@ revisions: loongarch64_unknown_none
//@ [loongarch64_unknown_none] compile-flags: --target loongarch64-unknown-none
//@ [loongarch64_unknown_none] needs-llvm-components: loongarch
//@ revisions: loongarch64_unknown_none_softfloat
//@ [loongarch64_unknown_none_softfloat] compile-flags: --target loongarch64-unknown-none-softfloat
//@ [loongarch64_unknown_none_softfloat] needs-llvm-components: loongarch
//@ revisions: m68k_unknown_linux_gnu
//@ [m68k_unknown_linux_gnu] compile-flags: --target m68k-unknown-linux-gnu
//@ [m68k_unknown_linux_gnu] needs-llvm-components: m68k
//@ revisions: m68k_unknown_none_elf
//@ [m68k_unknown_none_elf] compile-flags: --target m68k-unknown-none-elf
//@ [m68k_unknown_none_elf] needs-llvm-components: m68k
//@ revisions: mips64_openwrt_linux_musl
//@ [mips64_openwrt_linux_musl] compile-flags: --target mips64-openwrt-linux-musl
//@ [mips64_openwrt_linux_musl] needs-llvm-components: mips
//@ revisions: mips64_unknown_linux_gnuabi64
//@ [mips64_unknown_linux_gnuabi64] compile-flags: --target mips64-unknown-linux-gnuabi64
//@ [mips64_unknown_linux_gnuabi64] needs-llvm-components: mips
//@ revisions: mips64_unknown_linux_muslabi64
//@ [mips64_unknown_linux_muslabi64] compile-flags: --target mips64-unknown-linux-muslabi64
//@ [mips64_unknown_linux_muslabi64] needs-llvm-components: mips
//@ revisions: mips64el_unknown_linux_gnuabi64
//@ [mips64el_unknown_linux_gnuabi64] compile-flags: --target mips64el-unknown-linux-gnuabi64
//@ [mips64el_unknown_linux_gnuabi64] needs-llvm-components: mips
//@ revisions: mips64el_unknown_linux_muslabi64
//@ [mips64el_unknown_linux_muslabi64] compile-flags: --target mips64el-unknown-linux-muslabi64
//@ [mips64el_unknown_linux_muslabi64] needs-llvm-components: mips
//@ revisions: mips_unknown_linux_gnu
//@ [mips_unknown_linux_gnu] compile-flags: --target mips-unknown-linux-gnu
//@ [mips_unknown_linux_gnu] needs-llvm-components: mips
//@ revisions: mips_unknown_linux_musl
//@ [mips_unknown_linux_musl] compile-flags: --target mips-unknown-linux-musl
//@ [mips_unknown_linux_musl] needs-llvm-components: mips
//@ revisions: mips_unknown_linux_uclibc
//@ [mips_unknown_linux_uclibc] compile-flags: --target mips-unknown-linux-uclibc
//@ [mips_unknown_linux_uclibc] needs-llvm-components: mips
//@ revisions: mips_mti_none_elf
//@ [mips_mti_none_elf] compile-flags: --target mips-mti-none-elf
//@ [mips_mti_none_elf] needs-llvm-components: mips
//@ revisions: mipsel_mti_none_elf
//@ [mipsel_mti_none_elf] compile-flags: --target mipsel-mti-none-elf
//@ [mipsel_mti_none_elf] needs-llvm-components: mips
//@ revisions: mipsel_sony_psp
//@ [mipsel_sony_psp] compile-flags: --target mipsel-sony-psp
//@ [mipsel_sony_psp] needs-llvm-components: mips
//@ revisions: mipsel_sony_psx
//@ [mipsel_sony_psx] compile-flags: --target mipsel-sony-psx
//@ [mipsel_sony_psx] needs-llvm-components: mips
//@ revisions: mipsel_unknown_linux_gnu
//@ [mipsel_unknown_linux_gnu] compile-flags: --target mipsel-unknown-linux-gnu
//@ [mipsel_unknown_linux_gnu] needs-llvm-components: mips
//@ revisions: mipsel_unknown_linux_musl
//@ [mipsel_unknown_linux_musl] compile-flags: --target mipsel-unknown-linux-musl
//@ [mipsel_unknown_linux_musl] needs-llvm-components: mips
//@ revisions: mipsel_unknown_linux_uclibc
//@ [mipsel_unknown_linux_uclibc] compile-flags: --target mipsel-unknown-linux-uclibc
//@ [mipsel_unknown_linux_uclibc] needs-llvm-components: mips
//@ revisions: mipsel_unknown_netbsd
//@ [mipsel_unknown_netbsd] compile-flags: --target mipsel-unknown-netbsd
//@ [mipsel_unknown_netbsd] needs-llvm-components: mips
//@ revisions: mipsel_unknown_none
//@ [mipsel_unknown_none] compile-flags: --target mipsel-unknown-none
//@ [mipsel_unknown_none] needs-llvm-components: mips
//@ revisions: mipsisa32r6_unknown_linux_gnu
//@ [mipsisa32r6_unknown_linux_gnu] compile-flags: --target mipsisa32r6-unknown-linux-gnu
//@ [mipsisa32r6_unknown_linux_gnu] needs-llvm-components: mips
//@ revisions: mipsisa32r6el_unknown_linux_gnu
//@ [mipsisa32r6el_unknown_linux_gnu] compile-flags: --target mipsisa32r6el-unknown-linux-gnu
//@ [mipsisa32r6el_unknown_linux_gnu] needs-llvm-components: mips
//@ revisions: mipsisa64r6_unknown_linux_gnuabi64
//@ [mipsisa64r6_unknown_linux_gnuabi64] compile-flags: --target mipsisa64r6-unknown-linux-gnuabi64
//@ [mipsisa64r6_unknown_linux_gnuabi64] needs-llvm-components: mips
//@ revisions: mipsisa64r6el_unknown_linux_gnuabi64
//@ [mipsisa64r6el_unknown_linux_gnuabi64] compile-flags: --target mipsisa64r6el-unknown-linux-gnuabi64
//@ [mipsisa64r6el_unknown_linux_gnuabi64] needs-llvm-components: mips
//@ revisions: msp430_none_elf
//@ [msp430_none_elf] compile-flags: --target msp430-none-elf
//@ [msp430_none_elf] needs-llvm-components: msp430
//@ revisions: powerpc64_unknown_freebsd
//@ [powerpc64_unknown_freebsd] compile-flags: --target powerpc64-unknown-freebsd
//@ [powerpc64_unknown_freebsd] needs-llvm-components: powerpc
//@ revisions: powerpc64_unknown_linux_gnu
//@ [powerpc64_unknown_linux_gnu] compile-flags: --target powerpc64-unknown-linux-gnu
//@ [powerpc64_unknown_linux_gnu] needs-llvm-components: powerpc
//@ revisions: powerpc64_unknown_linux_musl
//@ [powerpc64_unknown_linux_musl] compile-flags: --target powerpc64-unknown-linux-musl
//@ [powerpc64_unknown_linux_musl] needs-llvm-components: powerpc
//@ revisions: powerpc64_unknown_openbsd
//@ [powerpc64_unknown_openbsd] compile-flags: --target powerpc64-unknown-openbsd
//@ [powerpc64_unknown_openbsd] needs-llvm-components: powerpc
//@ revisions: powerpc64_wrs_vxworks
//@ [powerpc64_wrs_vxworks] compile-flags: --target powerpc64-wrs-vxworks
//@ [powerpc64_wrs_vxworks] needs-llvm-components: powerpc
//@ revisions: powerpc64le_unknown_freebsd
//@ [powerpc64le_unknown_freebsd] compile-flags: --target powerpc64le-unknown-freebsd
//@ [powerpc64le_unknown_freebsd] needs-llvm-components: powerpc
//@ revisions: powerpc64le_unknown_linux_gnu
//@ [powerpc64le_unknown_linux_gnu] compile-flags: --target powerpc64le-unknown-linux-gnu
//@ [powerpc64le_unknown_linux_gnu] needs-llvm-components: powerpc
//@ revisions: powerpc64le_unknown_linux_musl
//@ [powerpc64le_unknown_linux_musl] compile-flags: --target powerpc64le-unknown-linux-musl
//@ [powerpc64le_unknown_linux_musl] needs-llvm-components: powerpc
//@ revisions: powerpc_unknown_freebsd
//@ [powerpc_unknown_freebsd] compile-flags: --target powerpc-unknown-freebsd
//@ [powerpc_unknown_freebsd] needs-llvm-components: powerpc
//@ revisions: powerpc_unknown_linux_gnu
//@ [powerpc_unknown_linux_gnu] compile-flags: --target powerpc-unknown-linux-gnu
//@ [powerpc_unknown_linux_gnu] needs-llvm-components: powerpc
//@ revisions: powerpc_unknown_linux_gnuspe
//@ [powerpc_unknown_linux_gnuspe] compile-flags: --target powerpc-unknown-linux-gnuspe
//@ [powerpc_unknown_linux_gnuspe] needs-llvm-components: powerpc
//@ revisions: powerpc_unknown_linux_musl
//@ [powerpc_unknown_linux_musl] compile-flags: --target powerpc-unknown-linux-musl
//@ [powerpc_unknown_linux_musl] needs-llvm-components: powerpc
//@ revisions: powerpc_unknown_linux_muslspe
//@ [powerpc_unknown_linux_muslspe] compile-flags: --target powerpc-unknown-linux-muslspe
//@ [powerpc_unknown_linux_muslspe] needs-llvm-components: powerpc
//@ revisions: powerpc_unknown_netbsd
//@ [powerpc_unknown_netbsd] compile-flags: --target powerpc-unknown-netbsd
//@ [powerpc_unknown_netbsd] needs-llvm-components: powerpc
//@ revisions: powerpc_unknown_openbsd
//@ [powerpc_unknown_openbsd] compile-flags: --target powerpc-unknown-openbsd
//@ [powerpc_unknown_openbsd] needs-llvm-components: powerpc
//@ revisions: powerpc_wrs_vxworks
//@ [powerpc_wrs_vxworks] compile-flags: --target powerpc-wrs-vxworks
//@ [powerpc_wrs_vxworks] needs-llvm-components: powerpc
//@ revisions: powerpc_wrs_vxworks_spe
//@ [powerpc_wrs_vxworks_spe] compile-flags: --target powerpc-wrs-vxworks-spe
//@ [powerpc_wrs_vxworks_spe] needs-llvm-components: powerpc
//@ revisions: riscv32_wrs_vxworks
//@ [riscv32_wrs_vxworks] compile-flags: --target riscv32-wrs-vxworks
//@ [riscv32_wrs_vxworks] needs-llvm-components: riscv
//@ revisions: riscv32e_unknown_none_elf
//@ [riscv32e_unknown_none_elf] compile-flags: --target riscv32e-unknown-none-elf
//@ [riscv32e_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32em_unknown_none_elf
//@ [riscv32em_unknown_none_elf] compile-flags: --target riscv32em-unknown-none-elf
//@ [riscv32em_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32emc_unknown_none_elf
//@ [riscv32emc_unknown_none_elf] compile-flags: --target riscv32emc-unknown-none-elf
//@ [riscv32emc_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32gc_unknown_linux_gnu
//@ [riscv32gc_unknown_linux_gnu] compile-flags: --target riscv32gc-unknown-linux-gnu
//@ [riscv32gc_unknown_linux_gnu] needs-llvm-components: riscv
//@ revisions: riscv32gc_unknown_linux_musl
//@ [riscv32gc_unknown_linux_musl] compile-flags: --target riscv32gc-unknown-linux-musl
//@ [riscv32gc_unknown_linux_musl] needs-llvm-components: riscv
//@ revisions: riscv32i_unknown_none_elf
//@ [riscv32i_unknown_none_elf] compile-flags: --target riscv32i-unknown-none-elf
//@ [riscv32i_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32im_risc0_zkvm_elf
//@ [riscv32im_risc0_zkvm_elf] compile-flags: --target riscv32im-risc0-zkvm-elf
//@ [riscv32im_risc0_zkvm_elf] needs-llvm-components: riscv
//@ revisions: riscv32im_unknown_none_elf
//@ [riscv32im_unknown_none_elf] compile-flags: --target riscv32im-unknown-none-elf
//@ [riscv32im_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32ima_unknown_none_elf
//@ [riscv32ima_unknown_none_elf] compile-flags: --target riscv32ima-unknown-none-elf
//@ [riscv32ima_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32imac_esp_espidf
//@ [riscv32imac_esp_espidf] compile-flags: --target riscv32imac-esp-espidf
//@ [riscv32imac_esp_espidf] needs-llvm-components: riscv
//@ revisions: riscv32imac_unknown_none_elf
//@ [riscv32imac_unknown_none_elf] compile-flags: --target riscv32imac-unknown-none-elf
//@ [riscv32imac_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32imac_unknown_xous_elf
//@ [riscv32imac_unknown_xous_elf] compile-flags: --target riscv32imac-unknown-xous-elf
//@ [riscv32imac_unknown_xous_elf] needs-llvm-components: riscv
//@ revisions: riscv32imafc_unknown_none_elf
//@ [riscv32imafc_unknown_none_elf] compile-flags: --target riscv32imafc-unknown-none-elf
//@ [riscv32imafc_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv32imafc_esp_espidf
//@ [riscv32imafc_esp_espidf] compile-flags: --target riscv32imafc-esp-espidf
//@ [riscv32imafc_esp_espidf] needs-llvm-components: riscv
//@ revisions: riscv32imc_esp_espidf
//@ [riscv32imc_esp_espidf] compile-flags: --target riscv32imc-esp-espidf
//@ [riscv32imc_esp_espidf] needs-llvm-components: riscv
//@ revisions: riscv32imc_unknown_none_elf
//@ [riscv32imc_unknown_none_elf] compile-flags: --target riscv32imc-unknown-none-elf
//@ [riscv32imc_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv64_linux_android
//@ [riscv64_linux_android] compile-flags: --target riscv64-linux-android
//@ [riscv64_linux_android] needs-llvm-components: riscv
//@ revisions: riscv64_wrs_vxworks
//@ [riscv64_wrs_vxworks] compile-flags: --target riscv64-wrs-vxworks
//@ [riscv64_wrs_vxworks] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_freebsd
//@ [riscv64gc_unknown_freebsd] compile-flags: --target riscv64gc-unknown-freebsd
//@ [riscv64gc_unknown_freebsd] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_fuchsia
//@ [riscv64gc_unknown_fuchsia] compile-flags: --target riscv64gc-unknown-fuchsia
//@ [riscv64gc_unknown_fuchsia] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_hermit
//@ [riscv64gc_unknown_hermit] compile-flags: --target riscv64gc-unknown-hermit
//@ [riscv64gc_unknown_hermit] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_linux_gnu
//@ [riscv64gc_unknown_linux_gnu] compile-flags: --target riscv64gc-unknown-linux-gnu
//@ [riscv64gc_unknown_linux_gnu] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_linux_musl
//@ [riscv64gc_unknown_linux_musl] compile-flags: --target riscv64gc-unknown-linux-musl
//@ [riscv64gc_unknown_linux_musl] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_netbsd
//@ [riscv64gc_unknown_netbsd] compile-flags: --target riscv64gc-unknown-netbsd
//@ [riscv64gc_unknown_netbsd] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_none_elf
//@ [riscv64gc_unknown_none_elf] compile-flags: --target riscv64gc-unknown-none-elf
//@ [riscv64gc_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_openbsd
//@ [riscv64gc_unknown_openbsd] compile-flags: --target riscv64gc-unknown-openbsd
//@ [riscv64gc_unknown_openbsd] needs-llvm-components: riscv
//@ revisions: riscv64imac_unknown_none_elf
//@ [riscv64imac_unknown_none_elf] compile-flags: --target riscv64imac-unknown-none-elf
//@ [riscv64imac_unknown_none_elf] needs-llvm-components: riscv
//@ revisions: s390x_unknown_linux_gnu
//@ [s390x_unknown_linux_gnu] compile-flags: --target s390x-unknown-linux-gnu
//@ [s390x_unknown_linux_gnu] needs-llvm-components: systemz
//@ revisions: s390x_unknown_linux_musl
//@ [s390x_unknown_linux_musl] compile-flags: --target s390x-unknown-linux-musl
//@ [s390x_unknown_linux_musl] needs-llvm-components: systemz
//@ revisions: sparc64_unknown_linux_gnu
//@ [sparc64_unknown_linux_gnu] compile-flags: --target sparc64-unknown-linux-gnu
//@ [sparc64_unknown_linux_gnu] needs-llvm-components: sparc
//@ revisions: sparc64_unknown_netbsd
//@ [sparc64_unknown_netbsd] compile-flags: --target sparc64-unknown-netbsd
//@ [sparc64_unknown_netbsd] needs-llvm-components: sparc
//@ revisions: sparc64_unknown_openbsd
//@ [sparc64_unknown_openbsd] compile-flags: --target sparc64-unknown-openbsd
//@ [sparc64_unknown_openbsd] needs-llvm-components: sparc
//@ revisions: sparc_unknown_linux_gnu
//@ [sparc_unknown_linux_gnu] compile-flags: --target sparc-unknown-linux-gnu
//@ [sparc_unknown_linux_gnu] needs-llvm-components: sparc
//@ revisions: sparc_unknown_none_elf
//@ [sparc_unknown_none_elf] compile-flags: --target sparc-unknown-none-elf
//@ [sparc_unknown_none_elf] needs-llvm-components: sparc
//@ revisions: sparcv9_sun_solaris
//@ [sparcv9_sun_solaris] compile-flags: --target sparcv9-sun-solaris
//@ [sparcv9_sun_solaris] needs-llvm-components: sparc
//@ revisions: thumbv4t_none_eabi
//@ [thumbv4t_none_eabi] compile-flags: --target thumbv4t-none-eabi
//@ [thumbv4t_none_eabi] needs-llvm-components: arm
//@ revisions: thumbv5te_none_eabi
//@ [thumbv5te_none_eabi] compile-flags: --target thumbv5te-none-eabi
//@ [thumbv5te_none_eabi] needs-llvm-components: arm
//@ revisions: thumbv6m_none_eabi
//@ [thumbv6m_none_eabi] compile-flags: --target thumbv6m-none-eabi
//@ [thumbv6m_none_eabi] needs-llvm-components: arm
//@ revisions: thumbv7em_none_eabi
//@ [thumbv7em_none_eabi] compile-flags: --target thumbv7em-none-eabi
//@ [thumbv7em_none_eabi] needs-llvm-components: arm
//@ revisions: thumbv7em_none_eabihf
//@ [thumbv7em_none_eabihf] compile-flags: --target thumbv7em-none-eabihf
//@ [thumbv7em_none_eabihf] needs-llvm-components: arm
//@ revisions: thumbv7m_none_eabi
//@ [thumbv7m_none_eabi] compile-flags: --target thumbv7m-none-eabi
//@ [thumbv7m_none_eabi] needs-llvm-components: arm
//@ revisions: thumbv7neon_linux_androideabi
//@ [thumbv7neon_linux_androideabi] compile-flags: --target thumbv7neon-linux-androideabi
//@ [thumbv7neon_linux_androideabi] needs-llvm-components: arm
//@ revisions: thumbv7neon_unknown_linux_gnueabihf
//@ [thumbv7neon_unknown_linux_gnueabihf] compile-flags: --target thumbv7neon-unknown-linux-gnueabihf
//@ [thumbv7neon_unknown_linux_gnueabihf] needs-llvm-components: arm
//@ revisions: thumbv7neon_unknown_linux_musleabihf
//@ [thumbv7neon_unknown_linux_musleabihf] compile-flags: --target thumbv7neon-unknown-linux-musleabihf
//@ [thumbv7neon_unknown_linux_musleabihf] needs-llvm-components: arm
//@ revisions: thumbv8m_base_none_eabi
//@ [thumbv8m_base_none_eabi] compile-flags: --target thumbv8m.base-none-eabi
//@ [thumbv8m_base_none_eabi] needs-llvm-components: arm
//@ revisions: thumbv8m_main_none_eabi
//@ [thumbv8m_main_none_eabi] compile-flags: --target thumbv8m.main-none-eabi
//@ [thumbv8m_main_none_eabi] needs-llvm-components: arm
//@ revisions: thumbv8m_main_none_eabihf
//@ [thumbv8m_main_none_eabihf] compile-flags: --target thumbv8m.main-none-eabihf
//@ [thumbv8m_main_none_eabihf] needs-llvm-components: arm
//@ revisions: wasm32_unknown_emscripten
//@ [wasm32_unknown_emscripten] compile-flags: --target wasm32-unknown-emscripten
//@ [wasm32_unknown_emscripten] needs-llvm-components: webassembly
//@ revisions: wasm32_unknown_unknown
//@ [wasm32_unknown_unknown] compile-flags: --target wasm32-unknown-unknown
//@ [wasm32_unknown_unknown] needs-llvm-components: webassembly
//@ revisions: wasm32v1_none
//@ [wasm32v1_none] compile-flags: --target wasm32v1-none
//@ [wasm32v1_none] needs-llvm-components: webassembly
//@ revisions: wasm32_wasip1
//@ [wasm32_wasip1] compile-flags: --target wasm32-wasip1
//@ [wasm32_wasip1] needs-llvm-components: webassembly
//@ revisions: wasm32_wasip1_threads
//@ [wasm32_wasip1_threads] compile-flags: --target wasm32-wasip1-threads
//@ [wasm32_wasip1_threads] needs-llvm-components: webassembly
//@ revisions: wasm32_wasip2
//@ [wasm32_wasip2] compile-flags: --target wasm32-wasip2
//@ [wasm32_wasip2] needs-llvm-components: webassembly
//@ revisions: wasm32_wali_linux_musl
//@ [wasm32_wali_linux_musl] compile-flags: --target wasm32-wali-linux-musl
//@ [wasm32_wali_linux_musl] needs-llvm-components: webassembly
//@ revisions: wasm64_unknown_unknown
//@ [wasm64_unknown_unknown] compile-flags: --target wasm64-unknown-unknown
//@ [wasm64_unknown_unknown] needs-llvm-components: webassembly
//@ revisions: x86_64_fortanix_unknown_sgx
//@ [x86_64_fortanix_unknown_sgx] compile-flags: --target x86_64-fortanix-unknown-sgx
//@ [x86_64_fortanix_unknown_sgx] needs-llvm-components: x86
//@ revisions: x86_64_linux_android
//@ [x86_64_linux_android] compile-flags: --target x86_64-linux-android
//@ [x86_64_linux_android] needs-llvm-components: x86
//@ revisions: x86_64_lynx_lynxos178
//@ [x86_64_lynx_lynxos178] compile-flags: --target x86_64-lynx-lynxos178
//@ [x86_64_lynx_lynxos178] needs-llvm-components: x86
//@ revisions: x86_64_pc_nto_qnx710
//@ [x86_64_pc_nto_qnx710] compile-flags: --target x86_64-pc-nto-qnx710
//@ [x86_64_pc_nto_qnx710] needs-llvm-components: x86
//@ revisions: x86_64_pc_nto_qnx710_iosock
//@ [x86_64_pc_nto_qnx710_iosock] compile-flags: --target x86_64-pc-nto-qnx710_iosock
//@ [x86_64_pc_nto_qnx710_iosock] needs-llvm-components: x86
//@ revisions: x86_64_pc_nto_qnx800
//@ [x86_64_pc_nto_qnx800] compile-flags: --target x86_64-pc-nto-qnx800
//@ [x86_64_pc_nto_qnx800] needs-llvm-components: x86
//@ revisions: x86_64_pc_solaris
//@ [x86_64_pc_solaris] compile-flags: --target x86_64-pc-solaris
//@ [x86_64_pc_solaris] needs-llvm-components: x86
//@ revisions: x86_64_unikraft_linux_musl
//@ [x86_64_unikraft_linux_musl] compile-flags: --target x86_64-unikraft-linux-musl
//@ [x86_64_unikraft_linux_musl] needs-llvm-components: x86
//@ revisions: x86_64_unknown_dragonfly
//@ [x86_64_unknown_dragonfly] compile-flags: --target x86_64-unknown-dragonfly
//@ [x86_64_unknown_dragonfly] needs-llvm-components: x86
//@ revisions: x86_64_unknown_freebsd
//@ [x86_64_unknown_freebsd] compile-flags: --target x86_64-unknown-freebsd
//@ [x86_64_unknown_freebsd] needs-llvm-components: x86
//@ revisions: x86_64_unknown_fuchsia
//@ [x86_64_unknown_fuchsia] compile-flags: --target x86_64-unknown-fuchsia
//@ [x86_64_unknown_fuchsia] needs-llvm-components: x86
//@ revisions: x86_64_unknown_haiku
//@ [x86_64_unknown_haiku] compile-flags: --target x86_64-unknown-haiku
//@ [x86_64_unknown_haiku] needs-llvm-components: x86
//@ revisions: x86_64_unknown_hurd_gnu
//@ [x86_64_unknown_hurd_gnu] compile-flags: --target x86_64-unknown-hurd-gnu
//@ [x86_64_unknown_hurd_gnu] needs-llvm-components: x86
//@ revisions: x86_64_unknown_hermit
//@ [x86_64_unknown_hermit] compile-flags: --target x86_64-unknown-hermit
//@ [x86_64_unknown_hermit] needs-llvm-components: x86
//@ revisions: x86_64_unknown_illumos
//@ [x86_64_unknown_illumos] compile-flags: --target x86_64-unknown-illumos
//@ [x86_64_unknown_illumos] needs-llvm-components: x86
//@ revisions: x86_64_unknown_l4re_uclibc
//@ [x86_64_unknown_l4re_uclibc] compile-flags: --target x86_64-unknown-l4re-uclibc
//@ [x86_64_unknown_l4re_uclibc] needs-llvm-components: x86
//@ revisions: x86_64_unknown_linux_gnu
//@ [x86_64_unknown_linux_gnu] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86_64_unknown_linux_gnu] needs-llvm-components: x86
//@ revisions: x86_64_unknown_linux_gnux32
//@ [x86_64_unknown_linux_gnux32] compile-flags: --target x86_64-unknown-linux-gnux32
//@ [x86_64_unknown_linux_gnux32] needs-llvm-components: x86
//@ revisions: x86_64_unknown_linux_musl
//@ [x86_64_unknown_linux_musl] compile-flags: --target x86_64-unknown-linux-musl
//@ [x86_64_unknown_linux_musl] needs-llvm-components: x86
//@ revisions: x86_64_unknown_linux_ohos
//@ [x86_64_unknown_linux_ohos] compile-flags: --target x86_64-unknown-linux-ohos
//@ [x86_64_unknown_linux_ohos] needs-llvm-components: x86
//@ revisions: x86_64_unknown_linux_none
//@ [x86_64_unknown_linux_none] compile-flags: --target x86_64-unknown-linux-none
//@ [x86_64_unknown_linux_none] needs-llvm-components: x86
//@ revisions: x86_64_unknown_netbsd
//@ [x86_64_unknown_netbsd] compile-flags: --target x86_64-unknown-netbsd
//@ [x86_64_unknown_netbsd] needs-llvm-components: x86
//@ revisions: x86_64_unknown_none
//@ [x86_64_unknown_none] compile-flags: --target x86_64-unknown-none
//@ [x86_64_unknown_none] needs-llvm-components: x86
//@ revisions: x86_64_unknown_openbsd
//@ [x86_64_unknown_openbsd] compile-flags: --target x86_64-unknown-openbsd
//@ [x86_64_unknown_openbsd] needs-llvm-components: x86
//@ revisions: x86_64_unknown_redox
//@ [x86_64_unknown_redox] compile-flags: --target x86_64-unknown-redox
//@ [x86_64_unknown_redox] needs-llvm-components: x86
//@ revisions: x86_64_unknown_trusty
//@ [x86_64_unknown_trusty] compile-flags: --target x86_64-unknown-trusty
//@ [x86_64_unknown_trusty] needs-llvm-components: x86
//@ revisions: x86_64_wrs_vxworks
//@ [x86_64_wrs_vxworks] compile-flags: --target x86_64-wrs-vxworks
//@ [x86_64_wrs_vxworks] needs-llvm-components: x86
//@ revisions: thumbv6m_nuttx_eabi
//@ [thumbv6m_nuttx_eabi] compile-flags: --target thumbv6m-nuttx-eabi
//@ [thumbv6m_nuttx_eabi] needs-llvm-components: arm
//@ revisions: thumbv7a_nuttx_eabi
//@ [thumbv7a_nuttx_eabi] compile-flags: --target thumbv7a-nuttx-eabi
//@ [thumbv7a_nuttx_eabi] needs-llvm-components: arm
//@ revisions: thumbv7a_nuttx_eabihf
//@ [thumbv7a_nuttx_eabihf] compile-flags: --target thumbv7a-nuttx-eabihf
//@ [thumbv7a_nuttx_eabihf] needs-llvm-components: arm
//@ revisions: thumbv7m_nuttx_eabi
//@ [thumbv7m_nuttx_eabi] compile-flags: --target thumbv7m-nuttx-eabi
//@ [thumbv7m_nuttx_eabi] needs-llvm-components: arm
//@ revisions: thumbv7em_nuttx_eabi
//@ [thumbv7em_nuttx_eabi] compile-flags: --target thumbv7em-nuttx-eabi
//@ [thumbv7em_nuttx_eabi] needs-llvm-components: arm
//@ revisions: thumbv7em_nuttx_eabihf
//@ [thumbv7em_nuttx_eabihf] compile-flags: --target thumbv7em-nuttx-eabihf
//@ [thumbv7em_nuttx_eabihf] needs-llvm-components: arm
//@ revisions: thumbv8m_base_nuttx_eabi
//@ [thumbv8m_base_nuttx_eabi] compile-flags: --target thumbv8m.base-nuttx-eabi
//@ [thumbv8m_base_nuttx_eabi] needs-llvm-components: arm
//@ revisions: thumbv8m_main_nuttx_eabi
//@ [thumbv8m_main_nuttx_eabi] compile-flags: --target thumbv8m.main-nuttx-eabi
//@ [thumbv8m_main_nuttx_eabi] needs-llvm-components: arm
//@ revisions: thumbv8m_main_nuttx_eabihf
//@ [thumbv8m_main_nuttx_eabihf] compile-flags: --target thumbv8m.main-nuttx-eabihf
//@ [thumbv8m_main_nuttx_eabihf] needs-llvm-components: arm
//@ revisions: riscv32imc_unknown_nuttx_elf
//@ [riscv32imc_unknown_nuttx_elf] compile-flags: --target riscv32imc-unknown-nuttx-elf
//@ [riscv32imc_unknown_nuttx_elf] needs-llvm-components: riscv
//@ revisions: riscv32imac_unknown_nuttx_elf
//@ [riscv32imac_unknown_nuttx_elf] compile-flags: --target riscv32imac-unknown-nuttx-elf
//@ [riscv32imac_unknown_nuttx_elf] needs-llvm-components: riscv
//@ revisions: riscv32imafc_unknown_nuttx_elf
//@ [riscv32imafc_unknown_nuttx_elf] compile-flags: --target riscv32imafc-unknown-nuttx-elf
//@ [riscv32imafc_unknown_nuttx_elf] needs-llvm-components: riscv
//@ revisions: riscv64imac_unknown_nuttx_elf
//@ [riscv64imac_unknown_nuttx_elf] compile-flags: --target riscv64imac-unknown-nuttx-elf
//@ [riscv64imac_unknown_nuttx_elf] needs-llvm-components: riscv
//@ revisions: riscv64gc_unknown_nuttx_elf
//@ [riscv64gc_unknown_nuttx_elf] compile-flags: --target riscv64gc-unknown-nuttx-elf
//@ [riscv64gc_unknown_nuttx_elf] needs-llvm-components: riscv
// FIXME: disabled since it requires a custom LLVM until the upstream LLVM adds support for the target (https://github.com/espressif/llvm-project/issues/4)
/*
    revisions: xtensa_esp32_none_elf
    [xtensa_esp32_none_elf] compile-flags: --target xtensa-esp32-none-elf
    [xtensa_esp32_none_elf] needs-llvm-components: xtensa
    revisions: xtensa_esp32_espidf
    [xtensa_esp32_espidf] compile-flags: --target xtensa-esp32s2-espidf
    [xtensa_esp32_espidf] needs-llvm-components: xtensa
    revisions: xtensa_esp32s2_none_elf
    [xtensa_esp32s2_none_elf] compile-flags: --target xtensa-esp32s2-none-elf
    [xtensa_esp32s2_none_elf] needs-llvm-components: xtensa
    revisions: xtensa_esp32s2_espidf
    [xtensa_esp32s2_espidf] compile-flags: --target xtensa-esp32s2-espidf
    [xtensa_esp32s2_espidf] needs-llvm-components: xtensa
    revisions: xtensa_esp32s3_none_elf
    [xtensa_esp32s3_none_elf] compile-flags: --target xtensa-esp32s3-none-elf
    [xtensa_esp32s3_none_elf] needs-llvm-components: xtensa
    revisions: xtensa_esp32s3_espidf
    [xtensa_esp32s3_espidf] compile-flags: --target xtensa-esp32s3-espidf
    [xtensa_esp32s3_espidf] needs-llvm-components: xtensa
*/
// Sanity-check that each target can produce assembly code.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// Force linkage to ensure code is actually generated
#[no_mangle]
pub fn test() -> u8 {
    42
}

// CHECK: .text
