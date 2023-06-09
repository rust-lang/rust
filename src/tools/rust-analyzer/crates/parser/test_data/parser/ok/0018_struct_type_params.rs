struct S1<T>;
struct S2<T>(u32);
struct S3<T> { u: u32 }

struct S4<>;
struct S5<'a>;
struct S6<'a:>;
struct S7<'a: 'b>;
struct S8<'a: 'b + >;
struct S9<'a: 'b + 'c>;
struct S10<'a,>;
struct S11<'a, 'b>;
struct S12<'a: 'b+, 'b: 'c,>;

struct S13<T>;
struct S14<T, U>;
struct S15<'a, T, U>;
