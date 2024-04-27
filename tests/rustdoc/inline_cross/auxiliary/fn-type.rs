pub type F = for<'z, 'a, '_unused> fn(&'z for<'b> fn(&'b str), &'a ()) -> &'a ();
