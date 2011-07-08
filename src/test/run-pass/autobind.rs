fn f[T](&T[] x) -> T {
    ret x.(0);
}

fn g(fn(&int[]) -> int act) -> int {
    ret act(~[1, 2, 3]);
}

fn main() {
    assert g(f) == 1;
    let fn(&str[]) -> str f1 = f;
    assert f1(~["x", "y", "z"]) == "x";
}
