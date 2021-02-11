// Type ascription doesn't lead to unsoundness

#![feature(type_ascription)]

fn main() {
    let arr = &[1u8, 2, 3];
    let ref x = arr: &[u8];
      //~^ ERROR type ascriptions are not
    let ref mut x = arr: &[u8];
      //~^ ERROR type ascriptions are not
    match arr: &[u8] {
      //~^ ERROR type ascriptions are not
        ref x => {}
    }
    let _len = (arr: &[u8]).len();
      //~^ ERROR type ascriptions are not
}
