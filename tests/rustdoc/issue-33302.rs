// Ensure constant and array length values are not taken from source
// code, which wreaks havoc with macros.

macro_rules! make {
    ($n:expr) => {
        pub struct S;

        // @has issue_33302/constant.CST.html \
        //        '//pre[@class="rust item-decl"]' 'pub const CST: i32'
        pub const CST: i32 = ($n * $n);
        // @has issue_33302/static.ST.html \
        //        '//pre[@class="rust item-decl"]' 'pub static ST: i32'
        pub static ST: i32 = ($n * $n);

        pub trait T<X> {
            fn ignore(_: &X) {}
            const C: X;
            // @has issue_33302/trait.T.html \
            //        '//pre[@class="rust item-decl"]' 'const D: i32'
            // @has - '//*[@id="associatedconstant.D"]' 'const D: i32'
            const D: i32 = ($n * $n);
        }

        // @has issue_33302/struct.S.html \
        //        '//*[@class="impl"]' 'impl T<[i32; 16]> for S'
        // @has - '//*[@id="associatedconstant.C"]' 'const C: [i32; 16]'
        // @has - '//*[@id="associatedconstant.D"]' 'const D: i32'
        impl T<[i32; ($n * $n)]> for S {
            const C: [i32; ($n * $n)] = [0; ($n * $n)];
        }

        // @has issue_33302/struct.S.html \
        //        '//*[@class="impl"]' 'impl T<[i32; 16]> for S'
        // @has - '//*[@id="associatedconstant.C-1"]' 'const C: (i32,)'
        // @has - '//*[@id="associatedconstant.D-1"]' 'const D: i32'
        impl T<(i32,)> for S {
            const C: (i32,) = ($n,);
        }

        // @has issue_33302/struct.S.html \
        //        '//*[@class="impl"]' 'impl T<(i32, i32)> for S'
        // @has - '//*[@id="associatedconstant.C-2"]' 'const C: (i32, i32)'
        // @has - '//*[@id="associatedconstant.D-2"]' 'const D: i32'
        impl T<(i32, i32)> for S {
            const C: (i32, i32) = ($n, $n);
            const D: i32 = ($n / $n);
        }
    };
}

make!(4);
