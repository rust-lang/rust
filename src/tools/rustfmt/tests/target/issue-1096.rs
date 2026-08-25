struct StructA<T> /* comment 1 */ {
    t: T,
}

struct StructB<T> /* comment 2 */;

struct StructC /* comment 3 */;

struct StructD /* comment 4 */ {
    t: usize,
}

struct StructE<T>
/* comment 5 */
where
    T: Clone,
{
    t: usize,
}

struct StructF
/* comment 6 */
where
    T: Clone,
{
    t: usize,
}

struct StructG<T>
/* comment 7 */
// why a line comment??
{
    t: T,
}

struct StructH<T>
/* comment 8 */
// why a line comment??
where
    T: Clone,
{
    t: T,
}

enum EnumA<T> /* comment 8 */ {
    Field(T),
}

enum EnumB /* comment 9 */ {
    Field,
}

// Issue 2781
struct StructX1<T>
// where
//     T: Clone
{
    inner: String,
}

struct StructX2<
    T,
    U: Iterator<Item = String>,
    V: Iterator<Item = String>,
    W: Iterator<Item = String>,
>
// where
//     T: Clone
{
    inner: String,
}
