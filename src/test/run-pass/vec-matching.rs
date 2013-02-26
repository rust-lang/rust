pub fn main() {
    let x = [1];
    match x {
        [_, _, _, _, _, .._] => ::core::util::unreachable(),
        [.._, _, _, _, _] => ::core::util::unreachable(),
        [_, .._, _, _] => ::core::util::unreachable(),
        [_, _] => ::core::util::unreachable(),
        [a] => {
            fail_unless!(a == 1);
        }
        [] => ::core::util::unreachable()
    }
}
