//! Run-time feature detection on RISC-V.

features! {
    @TARGET: riscv;
    @CFG: any(target_arch = "riscv32", target_arch = "riscv64");
    @MACRO_NAME: is_riscv_feature_detected;
    @MACRO_ATTRS:
    /// A macro to test at *runtime* whether instruction sets are available on
    /// RISC-V platforms.
    ///
    /// RISC-V standard defined the base sets and the extension sets.
    /// The base sets are RV32I, RV64I, RV32E or RV128I. Any RISC-V platform
    /// must support one base set and/or multiple extension sets.
    ///
    /// Any RISC-V standard instruction sets can be in state of either ratified,
    /// frozen or draft. The version and status of current standard instruction
    /// sets can be checked out from preface section of the [ISA manual].
    ///
    /// Platform may define and support their own custom instruction sets with
    /// ISA prefix X. These sets are highly platform specific and should be
    /// detected with their own platform support crates.
    ///
    /// # Unprivileged Specification
    ///
    /// The supported ratified RISC-V instruction sets are as follows:
    ///
    /// * RV32I: `"rv32i"`
    /// * Zifencei: `"zifencei"`
    /// * Zihintpause: `"zihintpause"`
    /// * RV64I: `"rv64i"`
    /// * M: `"m"`
    /// * A: `"a"`
    /// * Zicsr: `"zicsr"`
    /// * Zicntr: `"zicntr"`
    /// * Zihpm: `"zihpm"`
    /// * F: `"f"`
    /// * D: `"d"`
    /// * Q: `"q"`
    /// * C: `"c"`
    ///
    /// There's also bases and extensions marked as standard instruction set,
    /// but they are in frozen or draft state. These instruction sets are also
    /// reserved by this macro and can be detected in the future platforms.
    ///
    /// Frozen RISC-V instruction sets:
    ///
    /// * Zfinx: `"zfinx"`
    /// * Zdinx: `"zdinx"`
    /// * Zhinx: `"zhinx"`
    /// * Zhinxmin: `"zhinxmin"`
    /// * Ztso: `"ztso"`
    ///
    /// Draft RISC-V instruction sets:
    ///
    /// * RV32E: `"rv32e"`
    /// * RV128I: `"rv128i"`
    /// * Zfh: `"zfh"`
    /// * Zfhmin: `"zfhmin"`
    /// * B: `"b"`
    /// * J: `"j"`
    /// * P: `"p"`
    /// * V: `"v"`
    /// * Zam: `"zam"`
    ///
    /// Defined by Privileged Specification:
    ///
    /// * Supervisor: `"s"`
    /// * Svnapot: `"svnapot"`
    /// * Svpbmt: `"svpbmt"`
    /// * Svinval: `"svinval"`
    /// * Hypervisor: `"h"`
    ///
    /// # RISC-V Bit-Manipulation ISA-extensions
    ///
    /// This document defined the following extensions:
    ///
    /// * Zba: `"zba"`
    /// * Zbb: `"zbb"`
    /// * Zbc: `"zbc"`
    /// * Zbs: `"zbs"`
    ///
    /// # RISC-V Cryptography Extensions
    ///
    /// These extensions are defined in Volume I, Scalar & Entropy Source
    /// Instructions:
    ///
    /// * Zbkb: `"zbkb"`
    /// * Zbkc: `"zbkc"`
    /// * Zbkx: `"zbkx"`
    /// * Zknd: `"zknd"`
    /// * Zkne: `"zkne"`
    /// * Zknh: `"zknh"`
    /// * Zksed: `"zksed"`
    /// * Zksh: `"zksh"`
    /// * Zkr: `"zkr"`
    /// * Zkn: `"zkn"`
    /// * Zks: `"zks"`
    /// * Zk: `"zk"`
    /// * Zkt: `"zkt"`
    ///
    /// [ISA manual]: https://github.com/riscv/riscv-isa-manual/
    #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")]
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv32i: "rv32i";
    /// RV32I Base Integer Instruction Set
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zifencei: "zifencei";
    /// "Zifencei" Instruction-Fetch Fence
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zihintpause: "zihintpause";
    /// "Zihintpause" Pause Hint
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv64i: "rv64i";
    /// RV64I Base Integer Instruction Set
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] m: "m";
    /// "M" Standard Extension for Integer Multiplication and Division
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] a: "a";
    /// "A" Standard Extension for Atomic Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zicsr: "zicsr";
    /// "Zicsr", Control and Status Register (CSR) Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zicntr: "zicntr";
    /// "Zicntr", Standard Extension for Base Counters and Timers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zihpm: "zihpm";
    /// "Zihpm", Standard Extension for Hardware Performance Counters
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] f: "f";
    /// "F" Standard Extension for Single-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] d: "d";
    /// "D" Standard Extension for Double-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] q: "q";
    /// "Q" Standard Extension for Quad-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] c: "c";
    /// "C" Standard Extension for Compressed Instructions

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfinx: "zfinx";
    /// "Zfinx" Standard Extension for Single-Precision Floating-Point in Integer Registers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zdinx: "zdinx";
    /// "Zdinx" Standard Extension for Double-Precision Floating-Point in Integer Registers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zhinx: "zhinx";
    /// "Zhinx" Standard Extension for Half-Precision Floating-Point in Integer Registers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zhinxmin: "zhinxmin";
    /// "Zhinxmin" Standard Extension for Minimal Half-Precision Floating-Point in Integer Registers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] ztso: "ztso";
    /// "Ztso" Standard Extension for Total Store Ordering

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv32e: "rv32e";
    /// RV32E Base Integer Instruction Set
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv128i: "rv128i";
    /// RV128I Base Integer Instruction Set
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfh: "zfh";
    /// "Zfh" Standard Extension for 16-Bit Half-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfhmin: "zfhmin";
    /// "Zfhmin" Standard Extension for Minimal Half-Precision Floating-Point Support
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] b: "b";
    /// "B" Standard Extension for Bit Manipulation
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] j: "j";
    /// "J" Standard Extension for Dynamically Translated Languages
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] p: "p";
    /// "P" Standard Extension for Packed-SIMD Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] v: "v";
    /// "V" Standard Extension for Vector Operations
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zam: "zam";
    /// "Zam" Standard Extension for Misaligned Atomics

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] s: "s";
    /// Supervisor-Level ISA
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] svnapot: "svnapot";
    /// "Svnapot" Standard Extension for NAPOT Translation Contiguity
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] svpbmt: "svpbmt";
    /// "Svpbmt" Standard Extension for Page-Based Memory Types
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] svinval: "svinval";
    /// "Svinval" Standard Extension for Fine-Grained Address-Translation Cache Invalidation
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] h: "h";
    /// Hypervisor Extension

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zba: "zba";
    /// "Zba" Standard Extension for Address Generation Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zbb: "zbb";
    /// "Zbb" Standard Extension for Basic Bit-Manipulation
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zbc: "zbc";
    /// "Zbc" Standard Extension for Carry-less Multiplication
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zbs: "zbs";
    /// "Zbs" Standard Extension for Single-Bit instructions

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zbkb: "zbkb";
    /// "Zbkb" Standard Extension for Bitmanip instructions for Cryptography
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zbkc: "zbkc";
    /// "Zbkc" Standard Extension for Carry-less multiply instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zbkx: "zbkx";
    /// "Zbkx" Standard Extension for Crossbar permutation instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zknd: "zknd";
    /// "Zknd" Standard Extension for NIST Suite: AES Decryption
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zkne: "zkne";
    /// "Zkne" Standard Extension for NIST Suite: AES Encryption
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zknh: "zknh";
    /// "Zknh" Standard Extension for NIST Suite: Hash Function Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zksed: "zksed";
    /// "Zksed" Standard Extension for ShangMi Suite: SM4 Block Cipher Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zksh: "zksh";
    /// "Zksh" Standard Extension for ShangMi Suite: SM3 Hash Function Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zkr: "zkr";
    /// "Zkr" Standard Extension for Entropy Source Extension
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zkn: "zkn";
    /// "Zkn" Standard Extension for NIST Algorithm Suite
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zks: "zks";
    /// "Zks" Standard Extension for ShangMi Algorithm Suite
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zk: "zk";
    /// "Zk" Standard Extension for Standard scalar cryptography extension
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zkt: "zkt";
    /// "Zkt" Standard Extension for Data Independent Execution Latency
}
