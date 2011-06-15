

tag list[T] { cons(@T, @list[T]); nil; }

fn main() {
    let list[int] a =
        cons[int](@10, @cons[int](@12, @cons[int](@13, @nil[int])));
}