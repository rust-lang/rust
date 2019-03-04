impl TestTrait {
    fn foo_one_pre(/* Important comment1 */ self) {}

    fn foo_one_post(self /* Important comment1 */) {}

    fn foo_pre(/* Important comment1 */ self, /* Important comment2 */ a: i32) {}

    fn foo_post(self /* Important comment1 */, a: i32 /* Important comment2 */) {}

    fn bar_pre(/* Important comment1 */ &mut self, /* Important comment2 */ a: i32) {}

    fn bar_post(&mut self /* Important comment1 */, a: i32 /* Important comment2 */) {}

    fn baz_pre(
        /* Important comment1 */
        self: X<'a, 'b>,
        /* Important comment2 */
        a: i32,
    ) {
    }

    fn baz_post(
        self: X<'a, 'b>, /* Important comment1 */
        a: i32,          /* Important comment2 */
    ) {
    }

    fn baz_tree_pre(
        /* Important comment1 */
        self: X<'a, 'b>,
        /* Important comment2 */
        a: i32,
        /* Important comment3 */
        b: i32,
    ) {
    }

    fn baz_tree_post(
        self: X<'a, 'b>, /* Important comment1 */
        a: i32,          /* Important comment2 */
        b: i32,          /* Important comment3 */
    ) {
    }

    fn multi_line(
        self: X<'a, 'b>, /* Important comment1-1 */
        /* Important comment1-2 */
        a: i32, /* Important comment2 */
        b: i32, /* Important comment3 */
    ) {
    }

    fn two_line_comment(
        self: X<'a, 'b>, /* Important comment1-1
                         Important comment1-2 */
        a: i32, /* Important comment2 */
        b: i32, /* Important comment3 */
    ) {
    }

    fn no_first_line_comment(
        self: X<'a, 'b>,
        /* Important comment2 */ a: i32,
        /* Important comment3 */ b: i32,
    ) {
    }
}
