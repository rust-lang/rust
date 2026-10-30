//@ check-pass

macro_rules! wat {
    () => {
        #[allow(unused_attributes)]
        {
            #[allow(unregistered_tool::lint)]
            println!("boo");
        }
    };
}

fn main() {
    wat!();
}
