type r = {
    field: fn@()
};

fn main() {
    let i: r = {field: fn() { }};
}