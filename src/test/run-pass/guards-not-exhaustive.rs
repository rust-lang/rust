enum Q { R(Option<uint>) }

fn xyzzy(q: Q) -> uint {
    match q {
        R(S) if S.is_some() => { 0 }
        _ => 1
    }
}


pub fn main() {
    fail_unless!(xyzzy(R(Some(5))) == 0);
}
