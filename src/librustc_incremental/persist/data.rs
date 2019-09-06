//! The data that gets serialized and deserialized.

use rustc::dep_graph::{WorkProduct, WorkProductId};

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedWorkProduct {
    /// The node that produced the work-product.
    pub id: WorkProductId,

    /// The work-product data itself.
    pub work_product: WorkProduct,
}
