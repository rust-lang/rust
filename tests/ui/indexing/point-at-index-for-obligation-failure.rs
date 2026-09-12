fn main() {
    let a = std::collections::HashMap::<String,String>::new();
    let s = "hello";
    let _b = a[ //~ ERROR E0277
        &s
    ];
}
