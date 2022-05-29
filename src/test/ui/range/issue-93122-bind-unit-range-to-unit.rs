// run-rustfix

fn main() {
    let () = ()..; //~ ERROR: mismatched types
    let () = ..(); //~ ERROR: mismatched types
    let () = ..=(); //~ ERROR: mismatched types
    let () = ()..(); //~ ERROR: mismatched types
    let () = ()..=(); //~ ERROR: mismatched types
}
