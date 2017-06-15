// rustfmt-max_width: 80

fn foo() {
    // This is where it gets good
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
            // and now, casually call a method on this
        }
    }).map(MaybeCached::Cached)
}
