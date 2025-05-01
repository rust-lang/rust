const C<i32>: u32 = 0
where i32: Copy;
trait Foo {
    const C: i32 where i32: Copy;
}
