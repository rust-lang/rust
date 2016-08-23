fn test2() {
    0i128; //~ ERROR 128-bit integers are not stable
}

fn test2_2() {
    0u128; //~ ERROR 128-bit integers are not stable
}

