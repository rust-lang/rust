fn add(i: [int], m: [mut int], c: [const int]) {

    // Check that:
    //  (1) vectors of any two mutabilities can be added
    //  (2) result has mutability of lhs

   add(i + [3],
       m + [3],
       c + [3]);

   add(i + [mut 3],
       m + [mut 3],
       c + [mut 3]);

   add(i + i,
       m + i,
       c + i);

   add(i + m,
       m + m,
       c + m);

   add(i + c,
       m + c,
       c + c);

   add(m + [3], //! ERROR mismatched types
       m + [3],
       m + [3]);

   add(i + [3],
       i + [3], //! ERROR mismatched types
       i + [3]);

   add(c + [3], //! ERROR mismatched types
       c + [3], //! ERROR mismatched types
       c + [3]);

   add(m + [mut 3], //! ERROR mismatched types
       m + [mut 3],
       m + [mut 3]);

   add(i + [mut 3],
       i + [mut 3], //! ERROR mismatched types
       i + [mut 3]);

   add(c + [mut 3], //! ERROR mismatched types
       c + [mut 3], //! ERROR mismatched types
       c + [mut 3]);

   add(m + i, //! ERROR mismatched types
       m + i,
       m + i);

   add(i + i,
       i + i, //! ERROR mismatched types
       i + i);

   add(c + i, //! ERROR mismatched types
       c + i, //! ERROR mismatched types
       c + i);

   add(m + m, //! ERROR mismatched types
       m + m,
       m + m);

   add(i + m,
       i + m, //! ERROR mismatched types
       i + m);

   add(c + m, //! ERROR mismatched types
       c + m, //! ERROR mismatched types
       c + m);

   add(m + c, //! ERROR mismatched types
       m + c,
       m + c);

   add(i + c,
       i + c, //! ERROR mismatched types
       i + c);

   add(c + c, //! ERROR mismatched types
       c + c, //! ERROR mismatched types
       c + c);
}

fn main() {
}
