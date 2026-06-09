fn main() {
    Some(for _ in [1].into_iter() {});
    Some(loop { break; });
    Some(while true {});
}
