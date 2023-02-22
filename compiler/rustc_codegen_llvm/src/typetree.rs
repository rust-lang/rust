use rustc_middle::middle::typetree::{TypeTree, Kind};
use crate::llvm;

pub fn to_enzyme_typetree(tree: TypeTree, llvm_data_layout: &str, llcx: &llvm::Context) -> llvm::TypeTree {
    tree.0.iter().fold(llvm::TypeTree::new(), |obj, x| {
        let inner_tt = if x.child.0.is_empty() {
            let scalar = match x.kind {
                Kind::Integer => llvm::CConcreteType::DT_Integer,
                Kind::Float => llvm::CConcreteType::DT_Float,
                Kind::Double => llvm::CConcreteType::DT_Double,
                Kind::Pointer => llvm::CConcreteType::DT_Pointer,
                _ => unreachable!(),
            };

            llvm::TypeTree::from_type(scalar, llcx)
        } else {
            to_enzyme_typetree(x.child.clone(), llvm_data_layout, llcx)
        };
    
        obj.merge(inner_tt.shift(llvm_data_layout, 0, x.size as isize, x.offset as usize))
    })
}
