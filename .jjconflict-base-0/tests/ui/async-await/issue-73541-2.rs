//@ edition:2018

async fn c() {
    'a: loop {
        macro_rules! b {
            () => {
                continue 'a
                //~^ ERROR use of unreachable label `'a`
            }
        }

        async {
            loop {
                b!();
            }
        };
    }
}

fn main() { }
