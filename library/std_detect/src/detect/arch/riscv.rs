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
    /// [ISA manual]: https://riscv.org/specifications/ratified/
    ///
    /// # Platform-specific/agnostic Behavior and Availability
    ///
    /// Runtime detection depends on the platform-specific feature detection
    /// facility and its availability per feature is
    /// highly platform/version-specific.
    ///
    /// Still, a best-effort attempt is performed to enable subset/dependent
    /// features if a superset feature is enabled regardless of the platform.
    /// For instance, if the A extension (`"a"`) is enabled, its subsets (the
    /// Zalrsc and Zaamo extensions; `"zalrsc"` and `"zaamo"`) are also enabled.
    /// Likewise, if the F extension (`"f"`) is enabled, one of its dependencies
    /// (the Zicsr extension `"zicsr"`) is also enabled.
    ///
    /// # Unprivileged Specification
    ///
    /// The supported ratified RISC-V instruction sets are as follows:
    ///
    /// * RV32E: `"rv32e"`
    /// * RV32I: `"rv32i"`
    /// * RV64I: `"rv64i"`
    /// * A: `"a"`
    ///   * Zaamo: `"zaamo"`
    ///   * Zalrsc: `"zalrsc"`
    /// * B: `"b"`
    ///   * Zba: `"zba"`
    ///   * Zbb: `"zbb"`
    ///   * Zbs: `"zbs"`
    /// * C: `"c"`
    ///   * Zca: `"zca"`
    ///   * Zcd: `"zcd"` (if D is enabled)
    ///   * Zcf: `"zcf"` (if F is enabled on RV32)
    /// * D: `"d"`
    /// * F: `"f"`
    /// * M: `"m"`
    /// * Q: `"q"`
    /// * V: `"v"`
    ///   * Zve32x: `"zve32x"`
    ///   * Zve32f: `"zve32f"`
    ///   * Zve64x: `"zve64x"`
    ///   * Zve64f: `"zve64f"`
    ///   * Zve64d: `"zve64d"`
    /// * Zicbom: `"zicbom"`
    /// * Zicboz: `"zicboz"`
    /// * Zicntr: `"zicntr"`
    /// * Zicond: `"zicond"`
    /// * Zicsr: `"zicsr"`
    /// * Zifencei: `"zifencei"`
    /// * Zihintntl: `"zihintntl"`
    /// * Zihintpause: `"zihintpause"`
    /// * Zihpm: `"zihpm"`
    /// * Zimop: `"zimop"`
    /// * Zacas: `"zacas"`
    /// * Zawrs: `"zawrs"`
    /// * Zfa: `"zfa"`
    /// * Zfbfmin: `"zfbfmin"`
    /// * Zfh: `"zfh"`
    ///   * Zfhmin: `"zfhmin"`
    /// * Zfinx: `"zfinx"`
    /// * Zdinx: `"zdinx"`
    /// * Zhinx: `"zhinx"`
    ///   * Zhinxmin: `"zhinxmin"`
    /// * Zcb: `"zcb"`
    /// * Zcmop: `"zcmop"`
    /// * Zbc: `"zbc"`
    /// * Zbkb: `"zbkb"`
    /// * Zbkc: `"zbkc"`
    /// * Zbkx: `"zbkx"`
    /// * Zk: `"zk"`
    /// * Zkn: `"zkn"`
    ///   * Zknd: `"zknd"`
    ///   * Zkne: `"zkne"`
    ///   * Zknh: `"zknh"`
    /// * Zkr: `"zkr"`
    /// * Zks: `"zks"`
    ///   * Zksed: `"zksed"`
    ///   * Zksh: `"zksh"`
    /// * Zkt: `"zkt"`
    /// * Zvbb: `"zvbb"`
    /// * Zvbc: `"zvbc"`
    /// * Zvfbfmin: `"zvfbfmin"`
    /// * Zvfbfwma: `"zvfbfwma"`
    /// * Zvfh: `"zvfh"`
    ///   * Zvfhmin: `"zvfhmin"`
    /// * Zvkb: `"zvkb"`
    /// * Zvkg: `"zvkg"`
    /// * Zvkn: `"zvkn"`
    ///   * Zvkned: `"zvkned"`
    ///   * Zvknha: `"zvknha"`
    ///   * Zvknhb: `"zvknhb"`
    /// * Zvknc: `"zvknc"`
    /// * Zvkng: `"zvkng"`
    /// * Zvks: `"zvks"`
    ///   * Zvksed: `"zvksed"`
    ///   * Zvksh: `"zvksh"`
    /// * Zvksc: `"zvksc"`
    /// * Zvksg: `"zvksg"`
    /// * Zvkt: `"zvkt"`
    /// * Ztso: `"ztso"`
    ///
    /// There's also bases and extensions marked as standard instruction set,
    /// but they are in frozen or draft state. These instruction sets are also
    /// reserved by this macro and can be detected in the future platforms.
    ///
    /// Draft RISC-V instruction sets:
    ///
    /// * RV128I: `"rv128i"`
    /// * J: `"j"`
    /// * P: `"p"`
    /// * Zam: `"zam"`
    ///
    /// # Performance Hints
    ///
    /// The two features below define performance hints for unaligned
    /// scalar/vector memory accesses, respectively.  If enabled, it denotes that
    /// corresponding unaligned memory access is reasonably fast.
    ///
    /// * `"unaligned-scalar-mem"`
    ///   * Runtime detection requires Linux kernel version 6.4 or later.
    /// * `"unaligned-vector-mem"`
    ///   * Runtime detection requires Linux kernel version 6.13 or later.
    #[stable(feature = "riscv_ratified", since = "1.78.0")]

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv32i: "rv32i";
    without cfg check: true;
    /// RV32I Base Integer Instruction Set
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv32e: "rv32e";
    without cfg check: true;
    /// RV32E Base Integer Instruction Set
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv64i: "rv64i";
    without cfg check: true;
    /// RV64I Base Integer Instruction Set
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] rv128i: "rv128i";
    without cfg check: true;
    /// RV128I Base Integer Instruction Set

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] unaligned_scalar_mem: "unaligned-scalar-mem";
    /// Has reasonably performant unaligned scalar
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] unaligned_vector_mem: "unaligned-vector-mem";
    /// Has reasonably performant unaligned vector

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zicsr: "zicsr";
    /// "Zicsr" Extension for Control and Status Register (CSR) Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zicntr: "zicntr";
    /// "Zicntr" Extension for Base Counters and Timers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zihpm: "zihpm";
    /// "Zihpm" Extension for Hardware Performance Counters
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zifencei: "zifencei";
    /// "Zifencei" Extension for Instruction-Fetch Fence

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zihintntl: "zihintntl";
    /// "Zihintntl" Extension for Non-Temporal Locality Hints
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zihintpause: "zihintpause";
    /// "Zihintpause" Extension for Pause Hint
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zimop: "zimop";
    /// "Zimop" Extension for May-Be-Operations
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zicbom: "zicbom";
    /// "Zicbom" Extension for Cache-Block Management Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zicboz: "zicboz";
    /// "Zicboz" Extension for Cache-Block Zero Instruction
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zicond: "zicond";
    /// "Zicond" Extension for Integer Conditional Operations

    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] m: "m";
    /// "M" Extension for Integer Multiplication and Division

    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] a: "a";
    /// "A" Extension for Atomic Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zalrsc: "zalrsc";
    /// "Zalrsc" Extension for Load-Reserved/Store-Conditional Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zaamo: "zaamo";
    /// "Zaamo" Extension for Atomic Memory Operations
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zawrs: "zawrs";
    /// "Zawrs" Extension for Wait-on-Reservation-Set Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zacas: "zacas";
    /// "Zacas" Extension for Atomic Compare-and-Swap (CAS) Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zam: "zam";
    without cfg check: true;
    /// "Zam" Extension for Misaligned Atomics
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] ztso: "ztso";
    /// "Ztso" Extension for Total Store Ordering

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] f: "f";
    /// "F" Extension for Single-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] d: "d";
    /// "D" Extension for Double-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] q: "q";
    without cfg check: true;
    /// "Q" Extension for Quad-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfh: "zfh";
    /// "Zfh" Extension for Half-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfhmin: "zfhmin";
    /// "Zfhmin" Extension for Minimal Half-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfa: "zfa";
    /// "Zfa" Extension for Additional Floating-Point Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfbfmin: "zfbfmin";
    /// "Zfbfmin" Extension for Scalar BF16 Converts

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zfinx: "zfinx";
    /// "Zfinx" Extension for Single-Precision Floating-Point in Integer Registers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zdinx: "zdinx";
    /// "Zdinx" Extension for Double-Precision Floating-Point in Integer Registers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zhinx: "zhinx";
    /// "Zhinx" Extension for Half-Precision Floating-Point in Integer Registers
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zhinxmin: "zhinxmin";
    /// "Zhinxmin" Extension for Minimal Half-Precision Floating-Point in Integer Registers

    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] c: "c";
    /// "C" Extension for Compressed Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zca: "zca";
    /// "Zca" Compressed Instructions excluding Floating-Point Loads/Stores
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zcf: "zcf";
    without cfg check: true;
    /// "Zcf" Compressed Instructions for Single-Precision Floating-Point Loads/Stores on RV32
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zcd: "zcd";
    without cfg check: true;
    /// "Zcd" Compressed Instructions for Double-Precision Floating-Point Loads/Stores
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zcb: "zcb";
    /// "Zcb" Simple Code-size Saving Compressed Instructions
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zcmop: "zcmop";
    /// "Zcmop" Extension for Compressed May-Be-Operations

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] b: "b";
    /// "B" Extension for Bit Manipulation
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zba: "zba";
    /// "Zba" Extension for Address Generation
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zbb: "zbb";
    /// "Zbb" Extension for Basic Bit-Manipulation
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zbc: "zbc";
    /// "Zbc" Extension for Carry-less Multiplication
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zbs: "zbs";
    /// "Zbs" Extension for Single-Bit Instructions

    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zbkb: "zbkb";
    /// "Zbkb" Extension for Bit-Manipulation for Cryptography
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zbkc: "zbkc";
    /// "Zbkc" Extension for Carry-less Multiplication for Cryptography
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zbkx: "zbkx";
    /// "Zbkx" Extension for Crossbar Permutations
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zknd: "zknd";
    /// "Zknd" Cryptography Extension for NIST Suite: AES Decryption
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zkne: "zkne";
    /// "Zkne" Cryptography Extension for NIST Suite: AES Encryption
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zknh: "zknh";
    /// "Zknh" Cryptography Extension for NIST Suite: Hash Function Instructions
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zksed: "zksed";
    /// "Zksed" Cryptography Extension for ShangMi Suite: SM4 Block Cipher Instructions
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zksh: "zksh";
    /// "Zksh" Cryptography Extension for ShangMi Suite: SM3 Hash Function Instructions
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zkr: "zkr";
    /// "Zkr" Entropy Source Extension
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zkn: "zkn";
    /// "Zkn" Cryptography Extension for NIST Algorithm Suite
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zks: "zks";
    /// "Zks" Cryptography Extension for ShangMi Algorithm Suite
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zk: "zk";
    /// "Zk" Cryptography Extension for Standard Scalar Cryptography
    @FEATURE: #[stable(feature = "riscv_ratified", since = "1.78.0")] zkt: "zkt";
    /// "Zkt" Cryptography Extension for Data Independent Execution Latency

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] v: "v";
    /// "V" Extension for Vector Operations
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zve32x: "zve32x";
    /// "Zve32x" Vector Extension for Embedded Processors (32-bit+; Integer)
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zve32f: "zve32f";
    /// "Zve32f" Vector Extension for Embedded Processors (32-bit+; with Single-Precision Floating-Point)
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zve64x: "zve64x";
    /// "Zve64x" Vector Extension for Embedded Processors (64-bit+; Integer)
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zve64f: "zve64f";
    /// "Zve64f" Vector Extension for Embedded Processors (64-bit+; with Single-Precision Floating-Point)
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zve64d: "zve64d";
    /// "Zve64d" Vector Extension for Embedded Processors (64-bit+; with Double-Precision Floating-Point)
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvfh: "zvfh";
    /// "Zvfh" Vector Extension for Half-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvfhmin: "zvfhmin";
    /// "Zvfhmin" Vector Extension for Minimal Half-Precision Floating-Point
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvfbfmin: "zvfbfmin";
    /// "Zvfbfmin" Vector Extension for BF16 Converts
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvfbfwma: "zvfbfwma";
    /// "Zvfbfwma" Vector Extension for BF16 Widening Multiply-Add

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvbb: "zvbb";
    /// "Zvbb" Extension for Vector Basic Bit-Manipulation
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvbc: "zvbc";
    /// "Zvbc" Extension for Vector Carryless Multiplication
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvkb: "zvkb";
    /// "Zvkb" Extension for Vector Cryptography Bit-Manipulation
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvkg: "zvkg";
    /// "Zvkg" Cryptography Extension for Vector GCM/GMAC
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvkned: "zvkned";
    /// "Zvkned" Cryptography Extension for NIST Suite: Vector AES Block Cipher
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvknha: "zvknha";
    /// "Zvknha" Cryptography Extension for Vector SHA-2 Secure Hash (SHA-256)
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvknhb: "zvknhb";
    /// "Zvknhb" Cryptography Extension for Vector SHA-2 Secure Hash (SHA-256/512)
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvksed: "zvksed";
    /// "Zvksed" Cryptography Extension for ShangMi Suite: Vector SM4 Block Cipher
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvksh: "zvksh";
    /// "Zvksh" Cryptography Extension for ShangMi Suite: Vector SM3 Secure Hash
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvkn: "zvkn";
    /// "Zvkn" Cryptography Extension for NIST Algorithm Suite
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvknc: "zvknc";
    /// "Zvknc" Cryptography Extension for NIST Algorithm Suite with Carryless Multiply
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvkng: "zvkng";
    /// "Zvkng" Cryptography Extension for NIST Algorithm Suite with GCM
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvks: "zvks";
    /// "Zvks" Cryptography Extension for ShangMi Algorithm Suite
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvksc: "zvksc";
    /// "Zvksc" Cryptography Extension for ShangMi Algorithm Suite with Carryless Multiply
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvksg: "zvksg";
    /// "Zvksg" Cryptography Extension for ShangMi Algorithm Suite with GCM
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] zvkt: "zvkt";
    /// "Zvkt" Extension for Vector Data-Independent Execution Latency

    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] j: "j";
    without cfg check: true;
    /// "J" Extension for Dynamically Translated Languages
    @FEATURE: #[unstable(feature = "stdarch_riscv_feature_detection", issue = "111192")] p: "p";
    without cfg check: true;
    /// "P" Extension for Packed-SIMD Instructions
}
