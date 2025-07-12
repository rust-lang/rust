//@ jq .index[] | select(.name == "fst").inner.function.sig?.inputs[][0] == "(x, _)"
pub fn fst<X, Y>((x, _): (X, Y)) -> X {
    x
}

//@ jq .index[] | select(.name == "drop_int").inner.function.sig?.inputs[][0] == "_"
pub fn drop_int(_: i32) {}
