//! diagnostic test for #132749: ensure we pick a decent span and reason to blame for region errors
//! when failing to prove a region outlives 'static

struct Bytes(&'static [u8]);

fn deserialize_simple_string(buf: &[u8]) -> (Bytes, &[u8]) {
    //~^ NOTE let's call the lifetime of this reference `'1`
    let (s, rest) = buf.split_at(2);
    (Bytes(s), rest) //~ ERROR lifetime may not live long enough
    //~| NOTE this usage requires that `'1` must outlive `'static`
}

fn main() {}
