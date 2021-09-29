// edition:2018

struct Struct { }

impl Struct {
    async fn async_fn(self: &Struct, f: &u32) -> &u32 {
        f //~ ERROR: E0623
    }

    fn sync_fn(self: &Struct, f: &u32) -> &u32 {
        f //~ ERROR: E0623
    }
}

fn main() {}
