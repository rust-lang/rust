fn main() {
    let a: ~[int] = ~[];
    vec::each_ref(a, fn@(_x: &int) -> bool {
        //~^ ERROR not all control paths return a value
    });
}
