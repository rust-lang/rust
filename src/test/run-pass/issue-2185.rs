import iter::*;

fn main() {
    let range = bind uint::range(0u, 1000u, _);
    let filt = bind iter::filter(range, {|&&n: uint|
        n % 3u != 0u && n % 5u != 0u }, _);
    let sum = iter::foldl(filt, 0u) {|accum, &&n: uint| accum + n };

    io::println(#fmt("%u", sum));
}