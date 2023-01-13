// edition:2018

fn main() {
    'a: loop {
        async {
            loop {
                continue 'a
                //~^ ERROR use of unreachable label `'a`
            }
        };
    }
}
