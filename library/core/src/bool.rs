//! impl bool {}

impl bool {
    /// Returns `Some(t)` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// Arguments passed to `then_some` are eagerly evaluated; if you are
    /// passing the result of a function call, it is recommended to use
    /// [`then`], which is lazily evaluated.
    ///
    /// [`then`]: bool::then
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(false.then_some(0), None);
    /// assert_eq!(true.then_some(0), Some(0));
    /// ```
    ///
    /// ```
    /// let mut a = 0;
    /// let mut function_with_side_effects = || { a += 1; };
    ///
    /// true.then_some(function_with_side_effects());
    /// false.then_some(function_with_side_effects());
    ///
    /// // `a` is incremented twice because the value passed to `then_some` is
    /// // evaluated eagerly.
    /// assert_eq!(a, 2);
    /// ```
    #[stable(feature = "bool_to_option", since = "1.62.0")]
    #[inline]
    pub fn then_some<T>(self, t: T) -> Option<T> {
        if self { Some(t) } else { None }
    }

    /// Returns `Some(f())` if the `bool` is [`true`](../std/keyword.true.html),
    /// or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(false.then(|| 0), None);
    /// assert_eq!(true.then(|| 0), Some(0));
    /// ```
    ///
    /// ```
    /// let mut a = 0;
    ///
    /// true.then(|| { a += 1; });
    /// false.then(|| { a += 1; });
    ///
    /// // `a` is incremented once because the closure is evaluated lazily by
    /// // `then`.
    /// assert_eq!(a, 1);
    /// ```
    #[doc(alias = "then_with")]
    #[stable(feature = "lazy_bool_to_option", since = "1.50.0")]
    #[rustc_diagnostic_item = "bool_then"]
    #[inline]
    pub fn then<T, F: FnOnce() -> T>(self, f: F) -> Option<T> {
        if self { Some(f()) } else { None }
    }

    /// Returns either `true_val` or `false_val` depending on the value of
    /// `self`, with a hint to the compiler that `self` is unlikely
    /// to be correctly predicted by a CPU’s branch predictor.
    ///
    /// This method is functionally equivalent to
    /// ```ignore (this is just for illustrative purposes)
    /// fn select_unpredictable<T>(b: bool, true_val: T, false_val: T) -> T {
    ///     if b { true_val } else { false_val }
    /// }
    /// ```
    /// but might generate different assembly. In particular, on platforms with
    /// a conditional move or select instruction (like `cmov` on x86 or `csel`
    /// on ARM) the optimizer might use these instructions to avoid branches,
    /// which can benefit performance if the branch predictor is struggling
    /// with predicting `condition`, such as in an implementation of  binary
    /// search.
    ///
    /// Note however that this lowering is not guaranteed (on any platform) and
    /// should not be relied upon when trying to write constant-time code. Also
    /// be aware that this lowering might *decrease* performance if `condition`
    /// is well-predictable. It is advisable to perform benchmarks to tell if
    /// this function is useful.
    ///
    /// # Examples
    ///
    /// Distribute values evenly between two buckets:
    /// ```
    /// #![feature(select_unpredictable)]
    ///
    /// use std::hash::BuildHasher;
    ///
    /// fn append<H: BuildHasher>(hasher: &H, v: i32, bucket_one: &mut Vec<i32>, bucket_two: &mut Vec<i32>) {
    ///     let hash = hasher.hash_one(&v);
    ///     let bucket = (hash % 2 == 0).select_unpredictable(bucket_one, bucket_two);
    ///     bucket.push(v);
    /// }
    /// # let hasher = std::collections::hash_map::RandomState::new();
    /// # let mut bucket_one = Vec::new();
    /// # let mut bucket_two = Vec::new();
    /// # append(&hasher, 42, &mut bucket_one, &mut bucket_two);
    /// # assert_eq!(bucket_one.len() + bucket_two.len(), 1);
    /// ```
    #[inline(always)]
    #[unstable(feature = "select_unpredictable", issue = "133962")]
    pub fn select_unpredictable<T>(self, true_val: T, false_val: T) -> T {
        crate::intrinsics::select_unpredictable(self, true_val, false_val)
    }
}
