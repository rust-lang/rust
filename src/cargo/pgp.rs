use std;

import std::fs;
import std::run;

fn gpg(args: [str]) -> { status: int, out: str, err: str } {
    ret run::program_output("gpg", args);
}

fn signing_key() -> str {
    "
-----BEGIN PGP PUBLIC KEY BLOCK-----
Version: SKS 1.1.0

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
=2z6P
-----END PGP PUBLIC KEY BLOCK-----
"
}

fn signing_key_fp() -> str {
    "FE79 EDB0 3DEF B0D8 27D2  6C41 0B2D 6A28 3033 6376"
}

fn supported() -> bool {
    let r = gpg(["--version"]);
    r.status == 0
}

fn init(root: str) {
    let p = fs::connect(root, "gpg");
    if !fs::path_is_dir(p) {
        fs::make_dir(p, 0x1c0i32);
        let p = run::start_program("gpg", ["--homedir", p, "--import"]);
        p.input().write_str(signing_key());
        let s = p.finish();
        if s != 0 {
            fail "pgp init failed";
        }
    }
}

fn add(root: str, key: str) {
    let path = fs::connect(root, "gpg");
    let p = run::program_output("gpg", ["--homedir", path, "--import", key]);
    if p.status != 0 {
        fail "pgp add failed: " + p.out;
    }
}

fn verify(root: str, data: str, sig: str, keyfp: str) -> bool {
    let path = fs::connect(root, "gpg");
    let p = gpg(["--homedir", path, "--with-fingerprint", "--verify", sig,
                 data]);
    let res = "Primary key fingerprint: " + keyfp;
    for line in str::split(p.err, '\n' as u8) {
        if line == res {
            ret true;
        }
    }
    ret false;
}
