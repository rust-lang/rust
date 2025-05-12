extern "C" {
    thread_local! {
      static FOO: u32 = 0;
      //~^ error: extern items cannot be `const`
      //~| error: incorrect `static` inside `extern` block
    }
}

macro_rules! hello {
    ($name:ident) => {
        const $name: () = ();
    };
}

extern "C" {
    hello! { yes }
    //~^ error: extern items cannot be `const`
    //~| error: incorrect `static` inside `extern` block
}

fn main() {}
