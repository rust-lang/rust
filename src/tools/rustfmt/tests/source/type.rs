// rustfmt-normalize_comments: true
fn types() {
    let x: [ Vec   < _ > ] = [];
    let y:  * mut [ SomeType ; konst_funk() ] = expr();
    let z: (/*#digits*/ usize, /*exp*/ i16) = funk();
    let z: ( usize  /*#digits*/ , i16 /*exp*/ ) = funk();
}

struct F {
    f: extern "C" fn(x: u8, ... /* comment */),
    g: extern "C" fn(x: u8,/* comment */ ...),
    h: extern "C" fn(x: u8, ... ),
    i: extern "C" fn(x: u8, /* comment 4*/ y: String, // comment 3
                     z: Foo, /* comment */ .../* comment 2*/ ),
}

fn issue_1006(def_id_to_string: for<'a, 'b> unsafe fn(TyCtxt<'b, 'tcx, 'tcx>, DefId) -> String) {}

fn impl_trait_fn_1() -> impl Fn(i32) -> Option<u8> {}

fn impl_trait_fn_2<E>() -> impl Future<Item=&'a i64,Error=E> {}

fn issue_1234() {
    do_parse!(name: take_while1!(is_token) >> (Header))
}

// #2510
impl CombineTypes {
    pub fn pop_callback(
        &self,
        query_id: Uuid,
    ) -> Option<
        (
            ProjectId,
            Box<FnMut(&ProjectState, serde_json::Value, bool) -> () + Sync + Send>,
        ),
    > {
        self.query_callbacks()(&query_id)
    }
}

// #2859
pub fn do_something<'a, T: Trait1 + Trait2 + 'a>(&fooo: u32) -> impl Future<
    Item = (
        impl Future<Item = (
        ), Error =   SomeError> + 'a,
        impl Future<Item = (), Error = SomeError> + 'a,
impl Future<Item = (), Error = SomeError > + 'a,
    ),
    Error = SomeError,
    >
    +
    'a {
}

pub fn do_something<'a, T: Trait1 + Trait2 + 'a>(    &fooo: u32,
) -> impl Future<
    Item = (
impl Future<Item = (), Error = SomeError> + 'a,
        impl Future<Item = (), Error = SomeError> + 'a,
        impl Future<Item = (), Error = SomeError> + 'a,
    ),
    Error = SomeError,
    >
    + Future<
    Item = (
        impl Future<Item = (), Error = SomeError> + 'a,
impl Future<Item = (), Error = SomeError> + 'a,
        impl Future<Item = (), Error = SomeError> + 'a,
    ),
    Error = SomeError,
        >
    + Future<
    Item = (
        impl Future<Item = (), Error = SomeError> + 'a,
   impl Future<Item = (), Error = SomeError> + 'a,
        impl Future<Item = (), Error = SomeError> + 'a,
    ),
    Error = SomeError,
        >
    +
    'a + 'b +
    'c {
}

// #3051
token![impl];
token![ impl ];

// #3060
macro_rules! foo {
    ($foo_api: ty) => {
        type Target = ( $foo_api ) + 'static;
    }
}

type Target = ( FooAPI ) + 'static;

// #3137
fn foo<T>(t: T)
where
    T: ( FnOnce() -> () ) + Clone,
    U: ( FnOnce() -> () ) + 'static,
{
}

// #3117
fn issue3117() {
    {
        {
            {
                {
                    {
                        {
                            {
                                {
                                    let opt: &mut Option<MyLongTypeHere> =
                                        unsafe { &mut *self.future.get() };
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// #3139
fn issue3139() {
    assert_eq!(
        to_json_value(&None ::  <i32>).unwrap(),
        json!(  { "test": None  ::  <i32> }  )
    );
}

// #3180
fn foo(a: SomeLongComplexType, b: SomeOtherLongComplexType) -> Box<Future<Item = AnotherLongType, Error = ALongErrorType>> {
}

type MyFn = fn(a: SomeLongComplexType, b: SomeOtherLongComplexType,) -> Box<Future<Item = AnotherLongType, Error = ALongErrorType>>;

// Const bound

trait T: [   const ] Super {}

const fn not_quite_const<S: [  const  ]  T>() -> i32 { <S as T>::CONST }

impl     const T for U {}

fn apit(_: impl [   const ] T) {}

fn rpit() -> impl [  const] T { S }

pub struct Foo<T: Trait>(T);
impl<T:   [  const] Trait> Foo<T> {
    fn new(t: T) -> Self {
        Self(t)
    }
}

// #4357
type T = typeof(
1);
impl T for  .. {
}
