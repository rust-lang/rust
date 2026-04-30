// Regression reported in https://github.com/rust-lang/rust/pull/156016#discussion_r3453131612

#![feature(view_types)]

macro_rules! m {
    ($ty:ty) => {
        compile_error!("ty fragment matched a view type");
        //~^ ERROR ty fragment matched a view type
    };
    (&().{}) => {};
}

m!(&().{});

fn main() {}
