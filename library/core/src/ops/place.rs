#![expect(missing_docs, reason = "WIP")]

use crate::ptr::Pointee;

#[unstable(feature = "field_projections", issue = "145383")]
pub trait PlaceProxy {
    #[unstable(feature = "field_projections", issue = "145383")]
    type Target: ?Sized;
}

#[unstable(feature = "field_projections", issue = "145383")]
pub trait PlaceHandle: Sized {
    #[unstable(feature = "field_projections", issue = "145383")]
    type Target: ?Sized;
}

#[unstable(feature = "field_projections", issue = "145383")]
pub unsafe trait DerefPlace: PlaceHandle
where
    Self::Target: PlaceProxy,
{
    #[unstable(feature = "field_projections", issue = "145383")]
    type PointeeHandle: PlaceHandle<Target = <Self::Target as PlaceProxy>::Target>;

    #[unstable(feature = "field_projections", issue = "145383")]
    unsafe fn deref_place(self) -> Self::PointeeHandle;
}

#[unstable(feature = "field_projections", issue = "145383")]
pub unsafe trait ProjectPlace<S>: PlaceHandle
where
    S: Subplace<Source = Self::Target>,
{
    #[unstable(feature = "field_projections", issue = "145383")]
    type Projected: PlaceHandle<Target = S::Target>;

    #[unstable(feature = "field_projections", issue = "145383")]
    unsafe fn project_place(self, subplace: S) -> Self::Projected;
}

#[unstable(feature = "field_projections", issue = "145383")]
pub unsafe trait Subplace: Sized {
    #[unstable(feature = "field_projections", issue = "145383")]
    type Source: ?Sized;
    #[unstable(feature = "field_projections", issue = "145383")]
    type Target: ?Sized;

    #[unstable(feature = "field_projections", issue = "145383")]
    fn offset(
        self,
        metadata: <Self::Source as Pointee>::Metadata,
    ) -> (usize, <Self::Target as Pointee>::Metadata);
}
