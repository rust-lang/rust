impl TestTrait {
    fn foo_one(/* Important comment1 */
    self) {
    }

    fn foo(
        /* Important comment1 */
        self,
        /* Important comment2 */
        a: i32,
    ) {
    }

    fn bar(
            /* Important comment1 */
    &mut self,
        /* Important comment2 */
            a: i32,
    ) {
    }

    fn baz(
    /* Important comment1 */
            self: X< 'a ,  'b >,
            /* Important comment2 */
    a: i32,
    ) {
    }

    fn baz_tree(
    /* Important comment1 */

            self: X< 'a ,  'b >,
        /* Important comment2 */
        a: i32,
        /* Important comment3 */
        b: i32,
    ) {
    }
}
