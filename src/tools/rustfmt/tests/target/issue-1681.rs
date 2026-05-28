// rustfmt-max_width: 80

// We would like to surround closure body with block when overflowing the last
// argument of function call if the last argument has condition and without
// block it may go multi lines.
fn foo() {
    refmut_map_result(self.cache.borrow_mut(), |cache| {
        match cache.entry(cache_key) {
            Occupied(entry) => Ok(entry.into_mut()),
            Vacant(entry) => {
                let statement = {
                    let sql = try!(entry.key().sql(source));
                    prepare_fn(&sql)
                };

                Ok(entry.insert(try!(statement)))
            }
        }
    })
    .map(MaybeCached::Cached)
}
