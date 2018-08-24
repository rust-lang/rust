//! The data that we will serialize and deserialize.

use rustc::dep_graph::{WorkProduct, WorkProductId};

#[derive(Debug, RustcEncodable, RustcDecodable)]
pub struct SerializedWorkProduct {
    /// node that produced the work-product
    pub id: WorkProductId,

    /// work-product data itself
    pub work_product: WorkProduct,
}
