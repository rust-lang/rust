fn main() {
    if let 0 | 1 = 0 { //~ ERROR multiple patterns in `if let` and `while let` are unstable
        ;
    }
    while let 0 | 1 = 1 { //~ ERROR multiple patterns in `if let` and `while let` are unstable
        break;
    }
}
