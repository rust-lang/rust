type r = {
    field: fn@()
};

fn main() {
    fn f() {}
    let i: r = {field: f};
}