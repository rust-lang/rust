//@ edition: 2021


trait A {
    async fn e() {
        Ok(())
        //~^ ERROR mismatched types
    }
}

fn main() {}
