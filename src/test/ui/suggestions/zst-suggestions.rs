// compile-flags: --crate-type lib

fn g() {
    let () = ()..; //~ ERROR: mismatched types [E0308]
    let () = ..(); //~ ERROR: mismatched types [E0308]
    let () = ..=(); //~ ERROR: mismatched types [E0308]
    let () = ()..(); //~ ERROR: mismatched types [E0308]
    let () = ()..=(); //~ ERROR: mismatched types [E0308]
}
