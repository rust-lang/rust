//@ edition: 2021

async fn dont_suggest() -> i32 {
    if false {
        return Ok(6);
        //~^ ERROR mismatched types
    }

    5
}

async fn do_suggest() -> i32 {
    if false {
        let s = Ok(6);
        return s;
        //~^ ERROR mismatched types
    }

    5
}

fn main() {}
