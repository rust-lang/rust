// pretty-exact

fn f(f: fn@(int)) { f(10) }

fn main() {
    do f { |i| assert i == 10 }
}
