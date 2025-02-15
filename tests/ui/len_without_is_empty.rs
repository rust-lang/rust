#![warn(clippy::len_without_is_empty)]
#![allow(dead_code, unused)]

pub struct PubOne;

impl PubOne {
    pub fn len(&self) -> isize {
        //~^ len_without_is_empty

        1
    }
}

impl PubOne {
    // A second impl for this struct -- the error span shouldn't mention this.
    pub fn irrelevant(&self) -> bool {
        false
    }
}

// Identical to `PubOne`, but with an `allow` attribute on the impl complaining `len`.
pub struct PubAllowed;

#[allow(clippy::len_without_is_empty)]
impl PubAllowed {
    pub fn len(&self) -> isize {
        1
    }
}

// No `allow` attribute on this impl block, but that doesn't matter -- we only require one on the
// impl containing `len`.
impl PubAllowed {
    pub fn irrelevant(&self) -> bool {
        false
    }
}

pub struct PubAllowedFn;

impl PubAllowedFn {
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> isize {
        1
    }
}

#[allow(clippy::len_without_is_empty)]
pub struct PubAllowedStruct;

impl PubAllowedStruct {
    pub fn len(&self) -> isize {
        1
    }
}

pub trait PubTraitsToo {
    //~^ len_without_is_empty

    fn len(&self) -> isize;
}

impl PubTraitsToo for One {
    fn len(&self) -> isize {
        0
    }
}

pub struct HasIsEmpty;

impl HasIsEmpty {
    pub fn len(&self) -> isize {
        //~^ len_without_is_empty

        1
    }

    fn is_empty(&self) -> bool {
        false
    }
}

pub struct HasWrongIsEmpty;

impl HasWrongIsEmpty {
    pub fn len(&self) -> isize {
        //~^ len_without_is_empty

        1
    }

    pub fn is_empty(&self, x: u32) -> bool {
        false
    }
}

pub struct MismatchedSelf;

impl MismatchedSelf {
    pub fn len(self) -> isize {
        //~^ len_without_is_empty

        1
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

struct NotPubOne;

impl NotPubOne {
    pub fn len(&self) -> isize {
        // No error; `len` is pub but `NotPubOne` is not exported anyway.
        1
    }
}

struct One;

impl One {
    fn len(&self) -> isize {
        // No error; `len` is private; see issue #1085.
        1
    }
}

trait TraitsToo {
    fn len(&self) -> isize;
    // No error; `len` is private; see issue #1085.
}

impl TraitsToo for One {
    fn len(&self) -> isize {
        0
    }
}

struct HasPrivateIsEmpty;

impl HasPrivateIsEmpty {
    pub fn len(&self) -> isize {
        1
    }

    fn is_empty(&self) -> bool {
        false
    }
}

struct Wither;

pub trait WithIsEmpty {
    fn len(&self) -> isize;
    fn is_empty(&self) -> bool;
}

impl WithIsEmpty for Wither {
    fn len(&self) -> isize {
        1
    }

    fn is_empty(&self) -> bool {
        false
    }
}

pub trait Empty {
    fn is_empty(&self) -> bool;
}

pub trait InheritingEmpty: Empty {
    // Must not trigger `LEN_WITHOUT_IS_EMPTY`.
    fn len(&self) -> isize;
}

// This used to ICE.
pub trait Foo: Sized {}

pub trait DependsOnFoo: Foo {
    //~^ len_without_is_empty

    fn len(&mut self) -> usize;
}

// issue #1562
pub struct MultipleImpls;

impl MultipleImpls {
    pub fn len(&self) -> usize {
        1
    }
}

impl MultipleImpls {
    pub fn is_empty(&self) -> bool {
        false
    }
}

// issue #6958
pub struct OptionalLen;

impl OptionalLen {
    pub fn len(&self) -> Option<usize> {
        Some(0)
    }

    pub fn is_empty(&self) -> Option<bool> {
        Some(true)
    }
}

pub struct OptionalLen2;
impl OptionalLen2 {
    pub fn len(&self) -> Option<usize> {
        Some(0)
    }

    pub fn is_empty(&self) -> bool {
        true
    }
}

pub struct OptionalLen3;
impl OptionalLen3 {
    pub fn len(&self) -> usize {
        //~^ len_without_is_empty

        0
    }

    // should lint, len is not an option
    pub fn is_empty(&self) -> Option<bool> {
        None
    }
}

pub struct ResultLen;
impl ResultLen {
    pub fn len(&self) -> Result<usize, ()> {
        //~^ len_without_is_empty
        //~| result_unit_err

        Ok(0)
    }

    // Differing result types
    pub fn is_empty(&self) -> Option<bool> {
        Some(true)
    }
}

pub struct ResultLen2;
impl ResultLen2 {
    pub fn len(&self) -> Result<usize, ()> {
        //~^ result_unit_err

        Ok(0)
    }

    pub fn is_empty(&self) -> Result<bool, ()> {
        //~^ result_unit_err

        Ok(true)
    }
}

pub struct ResultLen3;
impl ResultLen3 {
    pub fn len(&self) -> Result<usize, ()> {
        //~^ result_unit_err

        Ok(0)
    }

    // Non-fallible result is ok.
    pub fn is_empty(&self) -> bool {
        true
    }
}

pub struct OddLenSig;
impl OddLenSig {
    // don't lint
    pub fn len(&self) -> bool {
        true
    }
}

// issue #6958
pub struct AsyncLen;
impl AsyncLen {
    async fn async_task(&self) -> bool {
        true
    }

    pub async fn len(&self) -> usize {
        usize::from(!self.async_task().await)
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

// issue #7232
pub struct AsyncLenWithoutIsEmpty;
impl AsyncLenWithoutIsEmpty {
    pub async fn async_task(&self) -> bool {
        true
    }

    pub async fn len(&self) -> usize {
        //~^ len_without_is_empty

        usize::from(!self.async_task().await)
    }
}

// issue #7232
pub struct AsyncOptionLenWithoutIsEmpty;
impl AsyncOptionLenWithoutIsEmpty {
    async fn async_task(&self) -> bool {
        true
    }

    pub async fn len(&self) -> Option<usize> {
        //~^ len_without_is_empty

        None
    }
}

// issue #7232
pub struct AsyncOptionLenNonIntegral;
impl AsyncOptionLenNonIntegral {
    // don't lint
    pub async fn len(&self) -> Option<String> {
        None
    }
}

// issue #7232
pub struct AsyncResultLenWithoutIsEmpty;
impl AsyncResultLenWithoutIsEmpty {
    async fn async_task(&self) -> bool {
        true
    }

    pub async fn len(&self) -> Result<usize, ()> {
        //~^ len_without_is_empty

        Err(())
    }
}

// issue #7232
pub struct AsyncOptionLen;
impl AsyncOptionLen {
    async fn async_task(&self) -> bool {
        true
    }

    pub async fn len(&self) -> Result<usize, ()> {
        Err(())
    }

    pub async fn is_empty(&self) -> bool {
        true
    }
}

pub struct AsyncLenSyncIsEmpty;
impl AsyncLenSyncIsEmpty {
    pub async fn len(&self) -> u32 {
        0
    }

    pub fn is_empty(&self) -> bool {
        true
    }
}

// issue #9520
pub struct NonStandardLen;
impl NonStandardLen {
    // don't lint
    pub fn len(&self, something: usize) -> usize {
        something
    }
}

// issue #9520
pub struct NonStandardLenAndIsEmptySignature;
impl NonStandardLenAndIsEmptySignature {
    // don't lint
    pub fn len(&self, something: usize) -> usize {
        something
    }

    pub fn is_empty(&self, something: usize) -> bool {
        something == 0
    }
}

// test case for #9520 with generics in the function signature
pub trait TestResource {
    type NonStandardSignatureWithGenerics: Copy;
    fn lookup_content(&self, item: Self::NonStandardSignatureWithGenerics) -> Result<Option<&[u8]>, String>;
}
pub struct NonStandardSignatureWithGenerics(u32);
impl NonStandardSignatureWithGenerics {
    pub fn is_empty<T, U>(self, resource: &T) -> bool
    where
        T: TestResource<NonStandardSignatureWithGenerics = U>,
        U: Copy + From<NonStandardSignatureWithGenerics>,
    {
        if let Ok(Some(content)) = resource.lookup_content(self.into()) {
            content.is_empty()
        } else {
            true
        }
    }

    // test case for #9520 with generics in the function signature
    pub fn len<T, U>(self, resource: &T) -> usize
    where
        T: TestResource<NonStandardSignatureWithGenerics = U>,
        U: Copy + From<NonStandardSignatureWithGenerics>,
    {
        if let Ok(Some(content)) = resource.lookup_content(self.into()) {
            content.len()
        } else {
            0_usize
        }
    }
}

pub struct DifferingErrors;
impl DifferingErrors {
    pub fn len(&self) -> Result<usize, u8> {
        Ok(0)
    }

    pub fn is_empty(&self) -> Result<bool, u16> {
        Ok(true)
    }
}

// Issue #11165
pub struct Aliased1;
pub type Alias1 = Aliased1;

impl Alias1 {
    pub fn len(&self) -> usize {
        todo!()
    }

    pub fn is_empty(&self) -> bool {
        todo!()
    }
}

pub struct Aliased2;
pub type Alias2 = Aliased2;
impl Alias2 {
    pub fn len(&self) -> usize {
        //~^ len_without_is_empty

        todo!()
    }
}

fn main() {}
