use crate::llvm;
use rustc_middle::middle::typetree::{Kind, TypeTree};

pub fn to_enzyme_typetree(
    tree: TypeTree,
    llvm_data_layout: &str,
    llcx: &llvm::Context,
) -> llvm::TypeTree {
    tree.0.iter().fold(llvm::TypeTree::new(), |obj, x| {
        let scalar = match x.kind {
            Kind::Integer => llvm::CConcreteType::DT_Integer,
            Kind::Float => llvm::CConcreteType::DT_Float,
            Kind::Double => llvm::CConcreteType::DT_Double,
            Kind::Pointer => llvm::CConcreteType::DT_Pointer,
            _ => panic!("Unknown kind {:?}", x.kind),
        };

        let tt = llvm::TypeTree::from_type(scalar, llcx).only(-1);

        let tt = if !x.child.0.is_empty() {
            let inner_tt = to_enzyme_typetree(x.child.clone(), llvm_data_layout, llcx);
            tt.merge(inner_tt.only(-1))
        } else {
            tt
        };

        if x.offset != -1 {
            obj.merge(tt.shift(llvm_data_layout, 0, x.size as isize, x.offset as usize))
        } else {
            obj.merge(tt)
        }
    })
}
