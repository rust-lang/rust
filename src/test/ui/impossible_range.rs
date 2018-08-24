// Make sure that invalid ranges generate an error during HIR lowering, not an ICE

pub fn main() {
    ..;
    0..;
    ..1;
    0..1;
    ..=; //~ERROR inclusive range with no end
         //~^HELP bounded at the end
}

fn _foo1() {
    ..=1;
    0..=1;
    0..=; //~ERROR inclusive range with no end
          //~^HELP bounded at the end
}
