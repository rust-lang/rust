// edition:2018
async fn f() -> i32 {
    if true { //~ ERROR if may be missing an else clause
        return 0;
    }
}

fn main() {}
