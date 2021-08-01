; RUN: %opt < %s %loadEnzyme -enzyme -enzyme-preopt=false -enzyme-loose-types -S | FileCheck %s

source_filename = "ld-temp.o"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char>::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char>::_Alloc_hider" = type { i8* }
%union.anon = type { i64, [8 x i8] }
%"class.Kripke::Core::DataStore" = type { %"class.std::map" }
%"class.std::map" = type { %"class.std::_Rb_tree" }
%"class.std::_Rb_tree" = type { %"struct.std::_Rb_tree<std::__cxx11::basic_string<char>, std::pair<const std::__cxx11::basic_string<char>, Kripke::Core::BaseVar *>, std::_Select1st<std::pair<const std::__cxx11::basic_string<char>, Kripke::Core::BaseVar *>>, std::less<std::__cxx11::basic_string<char>>>::_Rb_tree_impl" }
%"struct.std::_Rb_tree<std::__cxx11::basic_string<char>, std::pair<const std::__cxx11::basic_string<char>, Kripke::Core::BaseVar *>, std::_Select1st<std::pair<const std::__cxx11::basic_string<char>, Kripke::Core::BaseVar *>>, std::less<std::__cxx11::basic_string<char>>>::_Rb_tree_impl" = type { %"struct.std::_Rb_tree_key_compare", %"struct.std::_Rb_tree_header" }
%"struct.std::_Rb_tree_key_compare" = type { %"class.std::ios_base::Init" }
%"struct.std::_Rb_tree_header" = type { %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
%"struct.std::_Rb_tree_node" = type { %"struct.std::_Rb_tree_node_base", %"struct.__gnu_cxx::__aligned_membuf" }
%"struct.__gnu_cxx::__aligned_membuf" = type { [40 x i8] }
%"class.Kripke::Core::FieldStorage" = type { %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::Set"*, %"class.std::vector.16", %"class.std::vector.21.74" }
%"class.Kripke::Core::DomainVar" = type { %"class.Kripke::Core::BaseVar", %"class.std::vector.16", %"class.std::vector.16", %"class.std::vector.21" }
%"class.Kripke::Core::BaseVar" = type { i32 (...)**, %"class.Kripke::Core::DataStore"* }
%"class.std::vector.21" = type { %"struct.std::_Vector_base.22" }
%"struct.std::_Vector_base.22" = type { %"struct.std::_Vector_base<Kripke::SdomId, std::allocator<Kripke::SdomId>>::_Vector_impl" }
%"struct.std::_Vector_base<Kripke::SdomId, std::allocator<Kripke::SdomId>>::_Vector_impl" = type { %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"* }
%"class.Kripke::SdomId" = type { %"struct.RAJA::IndexValue" }
%"struct.RAJA::IndexValue" = type { i64 }
%"class.Kripke::Core::Set" = type { %"class.Kripke::Core::DomainVar", %"class.std::vector.16", %"class.std::vector.16", i64 }
%"class.std::vector.16" = type { %"struct.std::_Vector_base.17" }
%"struct.std::_Vector_base.17" = type { %"struct.std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl" }
%"struct.std::_Vector_base<unsigned long, std::allocator<unsigned long>>::_Vector_impl" = type { i64*, i64*, i64* }
%"class.std::vector.21.74" = type { %"struct.std::_Vector_base.22.73" }
%"struct.std::_Vector_base.22.73" = type { %"struct.std::_Vector_base<double *, std::allocator<double *>>::_Vector_impl" }
%"struct.std::_Vector_base<double *, std::allocator<double *>>::_Vector_impl" = type { double**, double**, double** }
%"class.Kripke::Core::Field.33" = type { %"class.Kripke::Core::FieldStorage", %"class.std::vector.34.107" }
%"class.std::vector.34.107" = type { %"struct.std::_Vector_base.35" }
%"struct.std::_Vector_base.35" = type { %"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::ZoneJ, Kripke::ZoneK>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::ZoneJ, Kripke::ZoneK>, -1>>>::_Vector_impl" }
%"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::ZoneJ, Kripke::ZoneK>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::ZoneJ, Kripke::ZoneK>, -1>>>::_Vector_impl" = type { %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"* }
%"struct.RAJA::TypedLayout.61" = type { %"struct.RAJA::detail::LayoutBase_impl.40" }
%"struct.RAJA::detail::LayoutBase_impl.40" = type { [4 x i64], [4 x i64], [4 x i64], [4 x i64] }
%"class.Kripke::Core::Field" = type { %"class.Kripke::Core::FieldStorage", %"class.std::vector.19" }
%"class.std::vector.19" = type { %"struct.std::_Vector_base.20" }
%"struct.std::_Vector_base.20" = type { %"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::Zone>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::Zone>, -1>>>::_Vector_impl" }
%"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::Zone>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Direction, Kripke::Group, Kripke::Zone>, -1>>>::_Vector_impl" = type { %"struct.RAJA::TypedLayout"*, %"struct.RAJA::TypedLayout"*, %"struct.RAJA::TypedLayout"* }
%"struct.RAJA::TypedLayout" = type { %"struct.RAJA::detail::LayoutBase_impl.4" }
%"struct.RAJA::detail::LayoutBase_impl.4" = type { [3 x i64], [3 x i64], [3 x i64], [3 x i64] }
%"struct.(anonymous namespace)::QuadraturePoint" = type { double, double, double, double, i32, i32, i32, i32 }
%"class.Kripke::Core::FieldStorage.242" = type { %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::Set"*, %"class.std::vector.16", %"class.std::vector.13" }
%"class.std::vector.13" = type { %"struct.std::_Vector_base.14" }
%"struct.std::_Vector_base.14" = type { %"struct.std::_Vector_base<Kripke::Legendre *, std::allocator<Kripke::Legendre *>>::_Vector_impl" }
%"struct.std::_Vector_base<Kripke::Legendre *, std::allocator<Kripke::Legendre *>>::_Vector_impl" = type { %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"** }
%"class.Kripke::Core::Field.246" = type { %"class.Kripke::Core::FieldStorage.242", %"class.std::vector.19.245" }
%"class.std::vector.19.245" = type { %"struct.std::_Vector_base.20.244" }
%"struct.std::_Vector_base.20.244" = type { %"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment>, -1>>>::_Vector_impl" }
%"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment>, -1>>>::_Vector_impl" = type { %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"* }
%"struct.RAJA::TypedLayout.47.255" = type { %"struct.RAJA::detail::LayoutBase_impl.24" }
%"struct.RAJA::detail::LayoutBase_impl.24" = type { [1 x i64], [1 x i64], [1 x i64], [1 x i64] }
%"class.Kripke::Core::Field.61" = type { %"class.Kripke::Core::FieldStorage", %"class.std::vector.62" }
%"class.std::vector.62" = type { %"struct.std::_Vector_base.63" }
%"struct.std::_Vector_base.63" = type { %"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment, Kripke::Direction>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment, Kripke::Direction>, -1>>>::_Vector_impl" }
%"struct.std::_Vector_base<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment, Kripke::Direction>, -1>, std::allocator<RAJA::TypedLayout<long, camp::tuple<Kripke::Moment, Kripke::Direction>, -1>>>::_Vector_impl" = type { %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"* }
%"struct.RAJA::TypedLayout.67" = type { %"struct.RAJA::detail::LayoutBase_impl.68" }
%"struct.RAJA::detail::LayoutBase_impl.68" = type { [2 x i64], [2 x i64], [2 x i64], [2 x i64] }
%"class.Kripke::Core::FieldStorage.49" = type { %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::Set"*, %"class.std::vector.16", %"class.std::vector.50" }
%"class.std::vector.50" = type { %"struct.std::_Vector_base.51" }
%"struct.std::_Vector_base.51" = type { %"struct.std::_Vector_base<int *, std::allocator<int *>>::_Vector_impl" }
%"struct.std::_Vector_base<int *, std::allocator<int *>>::_Vector_impl" = type { i32**, i32**, i32** }
%"class.Kripke::Core::Field.48.258" = type { %"class.Kripke::Core::FieldStorage.49", %"class.std::vector.19.245" }
%"class.Kripke::Core::Field.35" = type { %"class.Kripke::Core::FieldStorage", %"class.std::vector.19.245" }
%"class.Kripke::Core::FieldStorage.49.396" = type { %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::Set"*, %"class.std::vector.16", %"class.std::vector.50.395" }
%"class.std::vector.50.395" = type { %"struct.std::_Vector_base.51.394" }
%"struct.std::_Vector_base.51.394" = type { %"struct.std::_Vector_base<long *, std::allocator<long *>>::_Vector_impl" }
%"struct.std::_Vector_base<long *, std::allocator<long *>>::_Vector_impl" = type { i64**, i64**, i64** }
%"class.Kripke::Core::Field.48.397" = type { %"class.Kripke::Core::FieldStorage.49.396", %"class.std::vector.19.245" }
%class.anon.509 = type { %"struct.Kripke::ArchLayoutV"*, %"class.std::ios_base::Init"*, %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"*, %"class.Kripke::Core::Set"*, %"class.Kripke::Core::Set"*, %"class.Kripke::Core::Set"*, %"class.Kripke::Core::Field"*, %"class.Kripke::Core::Field"*, %"class.Kripke::Core::Field.33"*, %"class.Kripke::Core::Field.246"*, %"class.Kripke::Core::Field.48.258"*, %"class.Kripke::Core::Field.246"*, %"class.Kripke::Core::Field.35"*, %"class.Kripke::Core::Field.246"* }
%"struct.Kripke::ArchLayoutV" = type { i32, i32 }
%"class.Kripke::BlockTimer" = type { %"class.Kripke::Timing"*, %"class.std::__cxx11::basic_string" }
%"class.Kripke::Timing" = type { %"class.Kripke::Core::BaseVar", %"class.std::map" }
%"class.Kripke::Timer" = type { i8, double, i64, %"class.RAJA::Timer" }
%"class.RAJA::Timer" = type { %"class.RAJA::ChronoTimer" }
%"class.RAJA::ChronoTimer" = type { %"class.Kripke::SdomId", %"class.Kripke::SdomId", double }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl" }
%"struct.std::_Tuple_impl" = type { %"struct.std::_Head_base" }
%"struct.std::_Head_base" = type { %"class.std::__cxx11::basic_string"* }
%"struct.std::_Rb_tree_node.542" = type { %"struct.std::_Rb_tree_node_base", %"struct.__gnu_cxx::__aligned_membuf.541" }
%"struct.__gnu_cxx::__aligned_membuf.541" = type { [80 x i8] }
%"struct.std::pair" = type { %"class.std::__cxx11::basic_string", %"class.Kripke::Timer" }

$_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_PN6Kripke4Core7BaseVarEESt10_Select1stISC_ESt4lessIS5_ESaISC_EE8_M_eraseEPSt13_Rb_tree_nodeISC_E = comdat any

$_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElNS0_5__ops15_Iter_less_iterEEvT_SF_T0_T1_ = comdat any

$_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_RT0_ = comdat any

$_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_SF_RT0_ = comdat any

$_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_SF_SF_T0_ = comdat any

$_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEET_SF_SF_SF_T0_ = comdat any

$_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElS7_NS0_5__ops15_Iter_less_iterEEvT_T0_SG_T1_T2_ = comdat any

$_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElS7_NS0_5__ops14_Iter_less_valEEvT_T0_SG_T1_RT2_ = comdat any

$_ZN6Kripke4Core12FieldStorageIdED2Ev = comdat any

$_ZN6Kripke4Core9DomainVarD2Ev = comdat any

$_ZN6Kripke4Core9DomainVarD0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageIdED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEED0Ev = comdat any

$_ZN6Kripke4Core3SetD2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED0Ev = comdat any

$_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageIiED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageIiED0Ev = comdat any

$_ZN6Kripke4Core5FieldIiJNS_9DirectionEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIiJNS_9DirectionEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_9DirectionEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_8LegendreEED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_8LegendreEED0Ev = comdat any

$_ZN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_7MixElemEED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_7MixElemEED0Ev = comdat any

$_ZN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIiJNS_4ZoneEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIiJNS_4ZoneEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_7MixElemEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_7MixElemEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_8MaterialEED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_8MaterialEED0Ev = comdat any

$_ZN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_4ZoneEED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_4ZoneEED0Ev = comdat any

$_ZN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_4ZoneEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_4ZoneEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5ZoneKEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5ZoneKEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5ZoneJEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5ZoneJEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5ZoneIEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIdJNS_5ZoneIEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageIlED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageIlED0Ev = comdat any

$_ZN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEED0Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED2Ev = comdat any

$_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED0Ev = comdat any

$_ZN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEED0Ev = comdat any

$_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEED2Ev = comdat any

$_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEED0Ev = comdat any

$_ZN6Kripke4Core3SetD0Ev = comdat any

$_ZZN6Kripke8dispatchI14ScatteringSdomJRNS_6SdomIdES3_RNS_4Core3SetES6_S6_RNS4_5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEESC_RNS7_IdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESF_EEERNS7_INS_7MixElemEJSA_EEERNS7_IiJSA_EEERNS7_ISD_JSI_EEERNS7_IdJSI_EEERNS7_ISE_JS8_EEEEEEvNS_11ArchLayoutVERKT_DpOT0_ENKUlSU_E_clINS_16ArchT_SequentialEEEDaSU_ = comdat any

$_ZNK6Kripke14DispatchHelperINS_16ArchT_SequentialEEclINS_11LayoutT_DZGE14ScatteringSdomJRNS_6SdomIdES7_RNS_4Core3SetESA_SA_RNS8_5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEESG_RNSB_IdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESJ_EEERNSB_INS_7MixElemEJSE_EEERNSB_IiJSE_EEERNSB_ISH_JSM_EEERNSB_IdJSM_EEERNSB_ISI_JSC_EEEEEEvT_RKT0_DpOT1_ = comdat any

$_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEN6Kripke5TimerESt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_ = comdat any

$_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_ = comdat any

$_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_ = comdat any

$_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_ = comdat any

$_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_ = comdat any

$_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E = comdat any

$_ZTIN6Kripke4Core7BaseVarE = comdat any

$_ZTSN6Kripke4Core7BaseVarE = comdat any

$_ZTIN6Kripke4Core12FieldStorageIdEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageIdEE = comdat any

$_ZTIN6Kripke4Core9DomainVarE = comdat any

$_ZTSN6Kripke4Core9DomainVarE = comdat any

$_ZTIN6Kripke4Core14PartitionSpaceE = comdat any

$_ZTIN6Kripke10ArchLayoutE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageIdEE = comdat any

$_ZTVN6Kripke4Core9DomainVarE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE = comdat any

$_ZTSN6Kripke10ArchLayoutE = comdat any

$_ZTSN6Kripke4Core14PartitionSpaceE = comdat any

$_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE = comdat any

$_ZTIN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE = comdat any

$_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIiJNS_9DirectionEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageIiEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageIiEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageIiEE = comdat any

$_ZTIN6Kripke4Core5FieldIiJNS_9DirectionEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIiJNS_9DirectionEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_9DirectionEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_9DirectionEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_9DirectionEEEE = comdat any

$_ZTVN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageINS_8LegendreEEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageINS_8LegendreEEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageINS_8LegendreEEE = comdat any

$_ZTIN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE = comdat any

$_ZTSN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE = comdat any

$_ZTVN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageINS_7MixElemEEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageINS_7MixElemEEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageINS_7MixElemEEE = comdat any

$_ZTIN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE = comdat any

$_ZTSN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIiJNS_4ZoneEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIiJNS_4ZoneEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIiJNS_4ZoneEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_7MixElemEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_7MixElemEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_7MixElemEEEE = comdat any

$_ZTVN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageINS_8MaterialEEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageINS_8MaterialEEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageINS_8MaterialEEE = comdat any

$_ZTIN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE = comdat any

$_ZTSN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE = comdat any

$_ZTVN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageINS_4ZoneEEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageINS_4ZoneEEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageINS_4ZoneEEE = comdat any

$_ZTIN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE = comdat any

$_ZTSN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_4ZoneEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_4ZoneEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_4ZoneEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_5ZoneKEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_5ZoneKEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_5ZoneKEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_5ZoneJEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_5ZoneJEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_5ZoneJEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIdJNS_5ZoneIEEEE = comdat any

$_ZTIN6Kripke4Core5FieldIdJNS_5ZoneIEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIdJNS_5ZoneIEEEE = comdat any

$_ZTVN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageIlEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageIlEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageIlEE = comdat any

$_ZTIN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE = comdat any

$_ZTSN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE = comdat any

$_ZTVN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE = comdat any

$_ZTVN6Kripke4Core12FieldStorageINS_6SdomIdEEE = comdat any

$_ZTIN6Kripke4Core12FieldStorageINS_6SdomIdEEE = comdat any

$_ZTSN6Kripke4Core12FieldStorageINS_6SdomIdEEE = comdat any

$_ZTIN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE = comdat any

$_ZTSN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE = comdat any

$_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE = comdat any

$_ZTIN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE = comdat any

$_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE = comdat any

@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*
@_ZTIN6Kripke4Core7BaseVarE = internal constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([23 x i8], [23 x i8]* @_ZTSN6Kripke4Core7BaseVarE, i32 0, i32 0) }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTSN6Kripke4Core7BaseVarE = internal constant [23 x i8] c"N6Kripke4Core7BaseVarE\00", comdat, align 1
@_ZTIN6Kripke4Core12FieldStorageIdEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([32 x i8], [32 x i8]* @_ZTSN6Kripke4Core12FieldStorageIdEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageIdEE = internal constant [32 x i8] c"N6Kripke4Core12FieldStorageIdEE\00", comdat, align 1
@_ZTIN6Kripke4Core9DomainVarE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([25 x i8], [25 x i8]* @_ZTSN6Kripke4Core9DomainVarE, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core9DomainVarE = internal constant [25 x i8] c"N6Kripke4Core9DomainVarE\00", comdat, align 1
@_ZTIN6Kripke4Core14PartitionSpaceE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([31 x i8], [31 x i8]* @_ZTSN6Kripke4Core14PartitionSpaceE, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*) }, comdat, align 8
@_ZTIN6Kripke10ArchLayoutE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([22 x i8], [22 x i8]* @_ZTSN6Kripke10ArchLayoutE, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*) }, comdat, align 8
@_ZTIN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([72 x i8], [72 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE = internal constant [72 x i8] c"N6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EED0Ev to i8*)] }, comdat, align 8, !type !0, !type !1, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageIdEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage"*)* @_ZN6Kripke4Core12FieldStorageIdED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage"*)* @_ZN6Kripke4Core12FieldStorageIdED0Ev to i8*)] }, comdat, align 8, !type !0, !type !2, !type !3
@_ZTVN6Kripke4Core9DomainVarE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*), i8* bitcast (void (%"class.Kripke::Core::DomainVar"*)* @_ZN6Kripke4Core9DomainVarD2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::DomainVar"*)* @_ZN6Kripke4Core9DomainVarD0Ev to i8*)] }, comdat, align 8, !type !2, !type !3
@_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !4, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([70 x i8], [70 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE = internal constant [70 x i8] c"N6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !5, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([70 x i8], [70 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE = internal constant [70 x i8] c"N6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.33"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !6, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([70 x i8], [70 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE = internal constant [70 x i8] c"N6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field"*)* @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field"*)* @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !7, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([56 x i8], [56 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE = internal constant [56 x i8] c"N6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !8, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([59 x i8], [59 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE = internal constant [59 x i8] c"N6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE\00", comdat, align 1
@_ZTSN6Kripke10ArchLayoutE = internal constant [22 x i8] c"N6Kripke10ArchLayoutE\00", comdat, align 1
@_ZTSN6Kripke4Core14PartitionSpaceE = internal constant [31 x i8] c"N6Kripke4Core14PartitionSpaceE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEED0Ev to i8*)] }, comdat, align 8, !type !9, !type !10, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED0Ev to i8*)] }, comdat, align 8, !type !9, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @_ZTSN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE = internal constant [49 x i8] c"N6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([57 x i8], [57 x i8]* @_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE = internal constant [57 x i8] c"N6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.61"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.61"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !11, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([51 x i8], [51 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE = internal constant [51 x i8] c"N6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.61"*)* @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.61"*)* @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !12, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([51 x i8], [51 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE = internal constant [51 x i8] c"N6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIiJNS_9DirectionEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIiJNS_9DirectionEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.48.258"*)* @_ZN6Kripke4Core5FieldIiJNS_9DirectionEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.48.258"*)* @_ZN6Kripke4Core5FieldIiJNS_9DirectionEEED0Ev to i8*)] }, comdat, align 8, !type !13, !type !14, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageIiEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIiEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.49"*)* @_ZN6Kripke4Core12FieldStorageIiED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.49"*)* @_ZN6Kripke4Core12FieldStorageIiED0Ev to i8*)] }, comdat, align 8, !type !13, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageIiEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([32 x i8], [32 x i8]* @_ZTSN6Kripke4Core12FieldStorageIiEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageIiEE = internal constant [32 x i8] c"N6Kripke4Core12FieldStorageIiEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldIiJNS_9DirectionEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([40 x i8], [40 x i8]* @_ZTSN6Kripke4Core5FieldIiJNS_9DirectionEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIiEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIiJNS_9DirectionEEEE = internal constant [40 x i8] c"N6Kripke4Core5FieldIiJNS_9DirectionEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_9DirectionEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_9DirectionEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_9DirectionEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !15, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_9DirectionEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([40 x i8], [40 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_9DirectionEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_9DirectionEEEE = internal constant [40 x i8] c"N6Kripke4Core5FieldIdJNS_9DirectionEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEED0Ev to i8*)] }, comdat, align 8, !type !16, !type !17, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageINS_8LegendreEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_8LegendreEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_8LegendreEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_8LegendreEED0Ev to i8*)] }, comdat, align 8, !type !16, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageINS_8LegendreEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([44 x i8], [44 x i8]* @_ZTSN6Kripke4Core12FieldStorageINS_8LegendreEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageINS_8LegendreEEE = internal constant [44 x i8] c"N6Kripke4Core12FieldStorageINS_8LegendreEEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([49 x i8], [49 x i8]* @_ZTSN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_8LegendreEEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE = internal constant [49 x i8] c"N6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.61"*)* @_ZN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.61"*)* @_ZN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !18, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([45 x i8], [45 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE = internal constant [45 x i8] c"N6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEED0Ev to i8*)] }, comdat, align 8, !type !19, !type !20, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageINS_7MixElemEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_7MixElemEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_7MixElemEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_7MixElemEED0Ev to i8*)] }, comdat, align 8, !type !19, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageINS_7MixElemEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([43 x i8], [43 x i8]* @_ZTSN6Kripke4Core12FieldStorageINS_7MixElemEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageINS_7MixElemEEE = internal constant [43 x i8] c"N6Kripke4Core12FieldStorageINS_7MixElemEEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([46 x i8], [46 x i8]* @_ZTSN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_7MixElemEEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE = internal constant [46 x i8] c"N6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIiJNS_4ZoneEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIiJNS_4ZoneEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.48.258"*)* @_ZN6Kripke4Core5FieldIiJNS_4ZoneEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.48.258"*)* @_ZN6Kripke4Core5FieldIiJNS_4ZoneEEED0Ev to i8*)] }, comdat, align 8, !type !13, !type !21, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIiJNS_4ZoneEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @_ZTSN6Kripke4Core5FieldIiJNS_4ZoneEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIiEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIiJNS_4ZoneEEEE = internal constant [35 x i8] c"N6Kripke4Core5FieldIiJNS_4ZoneEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_7MixElemEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_7MixElemEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_7MixElemEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_7MixElemEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !22, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_7MixElemEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([38 x i8], [38 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_7MixElemEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_7MixElemEEEE = internal constant [38 x i8] c"N6Kripke4Core5FieldIdJNS_7MixElemEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEED0Ev to i8*)] }, comdat, align 8, !type !23, !type !24, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageINS_8MaterialEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_8MaterialEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_8MaterialEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_8MaterialEED0Ev to i8*)] }, comdat, align 8, !type !23, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageINS_8MaterialEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([44 x i8], [44 x i8]* @_ZTSN6Kripke4Core12FieldStorageINS_8MaterialEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageINS_8MaterialEEE = internal constant [44 x i8] c"N6Kripke4Core12FieldStorageINS_8MaterialEEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([50 x i8], [50 x i8]* @_ZTSN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_8MaterialEEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE = internal constant [50 x i8] c"N6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEED0Ev to i8*)] }, comdat, align 8, !type !25, !type !26, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageINS_4ZoneEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_4ZoneEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_4ZoneEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_4ZoneEED0Ev to i8*)] }, comdat, align 8, !type !25, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageINS_4ZoneEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([40 x i8], [40 x i8]* @_ZTSN6Kripke4Core12FieldStorageINS_4ZoneEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageINS_4ZoneEEE = internal constant [40 x i8] c"N6Kripke4Core12FieldStorageINS_4ZoneEEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([46 x i8], [46 x i8]* @_ZTSN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_4ZoneEEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE = internal constant [46 x i8] c"N6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_4ZoneEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_4ZoneEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_4ZoneEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_4ZoneEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !27, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_4ZoneEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_4ZoneEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_4ZoneEEEE = internal constant [35 x i8] c"N6Kripke4Core5FieldIdJNS_4ZoneEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_5ZoneKEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_5ZoneKEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_5ZoneKEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_5ZoneKEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !28, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_5ZoneKEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([36 x i8], [36 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_5ZoneKEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_5ZoneKEEEE = internal constant [36 x i8] c"N6Kripke4Core5FieldIdJNS_5ZoneKEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_5ZoneJEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_5ZoneJEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_5ZoneJEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_5ZoneJEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !29, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_5ZoneJEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([36 x i8], [36 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_5ZoneJEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_5ZoneJEEEE = internal constant [36 x i8] c"N6Kripke4Core5FieldIdJNS_5ZoneJEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIdJNS_5ZoneIEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_5ZoneIEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_5ZoneIEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.35"*)* @_ZN6Kripke4Core5FieldIdJNS_5ZoneIEEED0Ev to i8*)] }, comdat, align 8, !type !0, !type !30, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldIdJNS_5ZoneIEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([36 x i8], [36 x i8]* @_ZTSN6Kripke4Core5FieldIdJNS_5ZoneIEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIdEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIdJNS_5ZoneIEEEE = internal constant [36 x i8] c"N6Kripke4Core5FieldIdJNS_5ZoneIEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.48.397"*)* @_ZN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.48.397"*)* @_ZN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEED0Ev to i8*)] }, comdat, align 8, !type !31, !type !32, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageIlEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIlEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.49.396"*)* @_ZN6Kripke4Core12FieldStorageIlED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.49.396"*)* @_ZN6Kripke4Core12FieldStorageIlED0Ev to i8*)] }, comdat, align 8, !type !31, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageIlEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([32 x i8], [32 x i8]* @_ZTSN6Kripke4Core12FieldStorageIlEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageIlEE = internal constant [32 x i8] c"N6Kripke4Core12FieldStorageIlEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([44 x i8], [44 x i8]* @_ZTSN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageIlEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE = internal constant [44 x i8] c"N6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEED0Ev to i8*)] }, comdat, align 8, !type !33, !type !34, !type !2, !type !3
@_ZTVN6Kripke4Core12FieldStorageINS_6SdomIdEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_6SdomIdEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::FieldStorage.242"*)* @_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED0Ev to i8*)] }, comdat, align 8, !type !33, !type !2, !type !3
@_ZTIN6Kripke4Core12FieldStorageINS_6SdomIdEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([42 x i8], [42 x i8]* @_ZTSN6Kripke4Core12FieldStorageINS_6SdomIdEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core12FieldStorageINS_6SdomIdEEE = internal constant [42 x i8] c"N6Kripke4Core12FieldStorageINS_6SdomIdEEE\00", comdat, align 1
@_ZTIN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([54 x i8], [54 x i8]* @_ZTSN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_6SdomIdEEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE = internal constant [54 x i8] c"N6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEED2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Field.246"*)* @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEED0Ev to i8*)] }, comdat, align 8, !type !9, !type !35, !type !2, !type !3
@_ZTIN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([54 x i8], [54 x i8]* @_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE to i8*) }, comdat, align 8
@_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE = internal constant [54 x i8] c"N6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE\00", comdat, align 1
@_ZTVN6Kripke4Core3SetE = internal unnamed_addr constant { [6 x i8*] } { [6 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core3SetE to i8*), i8* bitcast (void (%"class.Kripke::Core::Set"*)* @_ZN6Kripke4Core3SetD2Ev to i8*), i8* bitcast (void (%"class.Kripke::Core::Set"*)* @_ZN6Kripke4Core3SetD0Ev to i8*), i8* bitcast (void ()* @__cxa_pure_virtual to i8*), i8* bitcast (i64 (%"class.Kripke::Core::Set"*, i64, i64)* @_ZNK6Kripke4Core3Set7dimSizeENS_6SdomIdEm to i8*)] }, align 8, !type !36, !type !37, !type !38, !type !2, !type !39, !type !40, !type !3, !type !41, !type !42
@_ZTIN6Kripke4Core3SetE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([19 x i8], [19 x i8]* @_ZTSN6Kripke4Core3SetE, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core9DomainVarE to i8*) }, align 8
@_ZTSN6Kripke4Core3SetE = internal constant [19 x i8] c"N6Kripke4Core3SetE\00", align 1
@.str.256 = private unnamed_addr constant [11 x i8] c"Scattering\00", align 1
@.str.3.259 = private unnamed_addr constant [7 x i8] c"timing\00", align 1
@.str.5.264 = private unnamed_addr constant [7 x i8] c"pspace\00", align 1
@.str.6.265 = private unnamed_addr constant [10 x i8] c"Set/Group\00", align 1
@.str.7.266 = private unnamed_addr constant [11 x i8] c"Set/Moment\00", align 1
@.str.9.267 = private unnamed_addr constant [4 x i8] c"phi\00", align 1
@.str.10.268 = private unnamed_addr constant [8 x i8] c"phi_out\00", align 1
@.str.11.269 = private unnamed_addr constant [10 x i8] c"data/sigs\00", align 1
@.str.12.270 = private unnamed_addr constant [16 x i8] c"zone_to_mixelem\00", align 1
@.str.13.271 = private unnamed_addr constant [20 x i8] c"zone_to_num_mixelem\00", align 1
@.str.14.272 = private unnamed_addr constant [20 x i8] c"mixelem_to_material\00", align 1
@.str.15.273 = private unnamed_addr constant [20 x i8] c"mixelem_to_fraction\00", align 1
@.str.16.274 = private unnamed_addr constant [19 x i8] c"moment_to_legendre\00", align 1
@_ZStL19piecewise_construct.282 = internal constant %"class.std::ios_base::Init" undef, align 1
@_ZTVN6Kripke6TimingE = internal unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke6TimingE to i8*), i8* bitcast (void (%"class.Kripke::Timing"*)* @_ZN6Kripke6TimingD2Ev to i8*), i8* bitcast (void (%"class.Kripke::Timing"*)* @_ZN6Kripke6TimingD0Ev to i8*)] }, align 8, !type !2, !type !43
@_ZTIN6Kripke6TimingE = internal constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([17 x i8], [17 x i8]* @_ZTSN6Kripke6TimingE, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*) }, align 8
@_ZTSN6Kripke6TimingE = internal constant [17 x i8] c"N6Kripke6TimingE\00", align 1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nofree nosync nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) #1

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #0

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture, i8* noalias nocapture readonly, i64, i1) #0

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPv(i8*) local_unnamed_addr #2

declare i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32), i64* nonnull align 8 dereferenceable(8), i64) local_unnamed_addr #3

; Function Attrs: nobuiltin nofree allocsize(0)
declare nonnull i8* @_Znwm(i64) local_unnamed_addr #4

define void @caller(%"class.Kripke::Core::DataStore"* nonnull %i59, %"class.Kripke::Core::DataStore"* nonnull %i60) local_unnamed_addr {
bb:
  call void @_Z17__enzyme_autodiffIJPN6Kripke4Core9DataStoreES3_mbEEvPvDpT_(i8* bitcast (void (%"class.Kripke::Core::DataStore"*)* @_ZN6Kripke6Kernel10scatteringERNS_4Core9DataStoreE to i8*), %"class.Kripke::Core::DataStore"* nonnull %i59, %"class.Kripke::Core::DataStore"* nonnull %i60)
  ret void
}

declare void @_Z17__enzyme_autodiffIJPN6Kripke4Core9DataStoreES3_mbEEvPvDpT_(i8*, %"class.Kripke::Core::DataStore"*, %"class.Kripke::Core::DataStore"*) local_unnamed_addr

; Function Attrs: argmemonly nofree nounwind readonly willreturn
declare i32 @memcmp(i8* nocapture, i8* nocapture, i64) local_unnamed_addr #5

; Function Attrs: nounwind readonly
declare i8* @__dynamic_cast(i8*, i8*, i8*, i64) local_unnamed_addr #6

; Function Attrs: nounwind
declare void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 zeroext, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* nonnull align 8 dereferenceable(32)) local_unnamed_addr #7

; Function Attrs: nounwind readonly willreturn
declare %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"*) local_unnamed_addr #8

; Function Attrs: nounwind readonly willreturn
declare %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"*) local_unnamed_addr #8

; Function Attrs: uwtable
define internal fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_PN6Kripke4Core7BaseVarEESt10_Select1stISC_ESt4lessIS5_ESaISC_EE8_M_eraseEPSt13_Rb_tree_nodeISC_E(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node"* %arg1) unnamed_addr #9 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = icmp eq %"struct.std::_Rb_tree_node"* %arg1, null
  br i1 %i, label %bb19, label %bb2

bb2:                                              ; preds = %bb16, %bb
  %i3 = phi %"struct.std::_Rb_tree_node"* [ %i9, %bb16 ], [ %arg1, %bb ]
  %i4 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i3, i64 0, i32 0, i32 3
  %i5 = bitcast %"struct.std::_Rb_tree_node_base"** %i4 to %"struct.std::_Rb_tree_node"**
  %i6 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i5, align 8, !tbaa !51
  tail call fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_PN6Kripke4Core7BaseVarEESt10_Select1stISC_ESt4lessIS5_ESaISC_EE8_M_eraseEPSt13_Rb_tree_nodeISC_E(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node"* %i6)
  %i7 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i3, i64 0, i32 0, i32 2
  %i8 = bitcast %"struct.std::_Rb_tree_node_base"** %i7 to %"struct.std::_Rb_tree_node"**
  %i9 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i8, align 8, !tbaa !57
  %i10 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i3, i64 0, i32 1
  %i11 = bitcast %"struct.__gnu_cxx::__aligned_membuf"* %i10 to i8**
  %i12 = load i8*, i8** %i11, align 8, !tbaa !58
  %i13 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i3, i64 0, i32 1, i32 0, i64 16
  %i14 = icmp eq i8* %i12, %i13
  br i1 %i14, label %bb16, label %bb15

bb15:                                             ; preds = %bb2
  tail call void @_ZdlPv(i8* %i12) #16
  br label %bb16

bb16:                                             ; preds = %bb15, %bb2
  %i17 = bitcast %"struct.std::_Rb_tree_node"* %i3 to i8*
  tail call void @_ZdlPv(i8* nonnull %i17) #16
  %i18 = icmp eq %"struct.std::_Rb_tree_node"* %i9, null
  br i1 %i18, label %bb19, label %bb2, !llvm.loop !62

bb19:                                             ; preds = %bb16, %bb
  ret void
}

; Function Attrs: uwtable
define internal fastcc void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElNS0_5__ops15_Iter_less_iterEEvT_SF_T0_T1_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* %arg1, i64 %arg2) unnamed_addr #9 comdat personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = alloca %"class.std::ios_base::Init", align 1
  %i3 = alloca %"class.std::ios_base::Init", align 1
  %i4 = ptrtoint %"class.std::__cxx11::basic_string"* %arg1 to i64
  %i5 = ptrtoint %"class.std::__cxx11::basic_string"* %arg to i64
  %i6 = sub i64 %i4, %i5
  %i7 = icmp sgt i64 %i6, 512
  br i1 %i7, label %bb8, label %bb34

bb8:                                              ; preds = %bb
  %i9 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 1
  br label %bb10

bb10:                                             ; preds = %bb25, %bb8
  %i11 = phi i64 [ %i6, %bb8 ], [ %i32, %bb25 ]
  %i12 = phi i64 [ %arg2, %bb8 ], [ %i27, %bb25 ]
  %i13 = phi %"class.std::__cxx11::basic_string"* [ %arg1, %bb8 ], [ %i30, %bb25 ]
  %i14 = icmp eq i64 %i12, 0
  br i1 %i14, label %bb15, label %bb25

bb15:                                             ; preds = %bb10
  %i16 = getelementptr inbounds %"class.std::ios_base::Init", %"class.std::ios_base::Init"* %i3, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %i16)
  %i17 = getelementptr inbounds %"class.std::ios_base::Init", %"class.std::ios_base::Init"* %i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %i17)
  call fastcc void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_RT0_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* %i13, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %i)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %i17)
  br label %bb18

bb18:                                             ; preds = %bb18, %bb15
  %i19 = phi %"class.std::__cxx11::basic_string"* [ %i20, %bb18 ], [ %i13, %bb15 ]
  %i20 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i19, i64 -1
  call fastcc void @_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_SF_RT0_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* nonnull %i20, %"class.std::__cxx11::basic_string"* nonnull %i20, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %i3)
  %i21 = ptrtoint %"class.std::__cxx11::basic_string"* %i20 to i64
  %i22 = sub i64 %i21, %i5
  %i23 = icmp sgt i64 %i22, 32
  br i1 %i23, label %bb18, label %bb24, !llvm.loop !65

bb24:                                             ; preds = %bb18
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %i16)
  br label %bb34

bb25:                                             ; preds = %bb10
  %i26 = lshr i64 %i11, 6
  %i27 = add nsw i64 %i12, -1
  %i28 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i26
  %i29 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i13, i64 -1
  tail call fastcc void @_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_SF_SF_T0_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* nonnull %i9, %"class.std::__cxx11::basic_string"* %i28, %"class.std::__cxx11::basic_string"* nonnull %i29)
  %i30 = tail call fastcc %"class.std::__cxx11::basic_string"* @_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEET_SF_SF_SF_T0_(%"class.std::__cxx11::basic_string"* nonnull %i9, %"class.std::__cxx11::basic_string"* %i13, %"class.std::__cxx11::basic_string"* %arg)
  tail call fastcc void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElNS0_5__ops15_Iter_less_iterEEvT_SF_T0_T1_(%"class.std::__cxx11::basic_string"* %i30, %"class.std::__cxx11::basic_string"* %i13, i64 %i27)
  %i31 = ptrtoint %"class.std::__cxx11::basic_string"* %i30 to i64
  %i32 = sub i64 %i31, %i5
  %i33 = icmp sgt i64 %i32, 512
  br i1 %i33, label %bb10, label %bb34, !llvm.loop !66

bb34:                                             ; preds = %bb25, %bb24, %bb
  ret void
}

; Function Attrs: uwtable
define internal fastcc void @_ZSt11__make_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_RT0_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* %arg1, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg2) unnamed_addr #9 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = alloca %"class.std::__cxx11::basic_string", align 8
  %i3 = alloca %"class.std::__cxx11::basic_string", align 8
  %i4 = ptrtoint %"class.std::__cxx11::basic_string"* %arg1 to i64
  %i5 = ptrtoint %"class.std::__cxx11::basic_string"* %arg to i64
  %i6 = sub i64 %i4, %i5
  %i7 = ashr exact i64 %i6, 5
  %i8 = icmp slt i64 %i6, 64
  br i1 %i8, label %bb68, label %bb9

bb9:                                              ; preds = %bb
  %i10 = add nsw i64 %i7, -2
  %i11 = sdiv i64 %i10, 2
  %i12 = bitcast %"class.std::__cxx11::basic_string"* %i to i8*
  %i13 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 2
  %i14 = bitcast %"class.std::__cxx11::basic_string"* %i to %union.anon**
  %i15 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 0, i32 0
  %i16 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 2, i32 0
  %i17 = bitcast %union.anon* %i13 to i8*
  %i18 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 1
  %i19 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i3, i64 0, i32 2
  %i20 = bitcast %"class.std::__cxx11::basic_string"* %i3 to %union.anon**
  %i21 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i3, i64 0, i32 0, i32 0
  %i22 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i3, i64 0, i32 2, i32 0
  %i23 = bitcast %union.anon* %i19 to i8*
  %i24 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i3, i64 0, i32 1
  br label %bb25

bb25:                                             ; preds = %bb57, %bb9
  %i26 = phi i64 [ %i11, %bb9 ], [ %i53, %bb57 ]
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i12) #16
  %i27 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i26
  store %union.anon* %i13, %union.anon** %i14, align 8, !tbaa !67
  %i28 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i27, i64 0, i32 0, i32 0
  %i29 = load i8*, i8** %i28, align 8, !tbaa !58
  %i30 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i26, i32 2
  %i31 = bitcast %union.anon* %i30 to i8*
  %i32 = icmp eq i8* %i29, %i31
  br i1 %i32, label %bb33, label %bb34

bb33:                                             ; preds = %bb25
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %i17, i8* nonnull align 8 dereferenceable(16) %i29, i64 16, i1 false) #16
  br label %bb37

bb34:                                             ; preds = %bb25
  store i8* %i29, i8** %i15, align 8, !tbaa !58
  %i35 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i26, i32 2, i32 0
  %i36 = load i64, i64* %i35, align 8, !tbaa !68
  store i64 %i36, i64* %i16, align 8, !tbaa !68
  br label %bb37

bb37:                                             ; preds = %bb34, %bb33
  %i38 = phi i8* [ %i17, %bb33 ], [ %i29, %bb34 ]
  %i39 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i26, i32 1
  %i40 = load i64, i64* %i39, align 8, !tbaa !69
  %i41 = bitcast %"class.std::__cxx11::basic_string"* %i27 to %union.anon**
  store %union.anon* %i30, %union.anon** %i41, align 8, !tbaa !58
  store i64 0, i64* %i39, align 8, !tbaa !69
  store i8 0, i8* %i31, align 8, !tbaa !68
  store %union.anon* %i19, %union.anon** %i20, align 8, !tbaa !67
  %i42 = icmp eq i8* %i38, %i17
  br i1 %i42, label %bb43, label %bb44

bb43:                                             ; preds = %bb37
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %i23, i8* nonnull align 8 dereferenceable(16) %i17, i64 16, i1 false) #16
  br label %bb46

bb44:                                             ; preds = %bb37
  store i8* %i38, i8** %i21, align 8, !tbaa !58
  %i45 = load i64, i64* %i16, align 8, !tbaa !68
  store i64 %i45, i64* %i22, align 8, !tbaa !68
  br label %bb46

bb46:                                             ; preds = %bb44, %bb43
  store i64 %i40, i64* %i24, align 8, !tbaa !69
  store %union.anon* %i13, %union.anon** %i14, align 8, !tbaa !58
  store i64 0, i64* %i18, align 8, !tbaa !69
  store i8 0, i8* %i17, align 8, !tbaa !68
  call fastcc void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElS7_NS0_5__ops15_Iter_less_iterEEvT_T0_SG_T1_T2_(%"class.std::__cxx11::basic_string"* nonnull %arg, i64 %i26, i64 %i7, %"class.std::__cxx11::basic_string"* nonnull %i3)
  %i48 = load i8*, i8** %i21, align 8, !tbaa !58
  %i49 = icmp eq i8* %i48, %i23
  br i1 %i49, label %bb51, label %bb50

bb50:                                             ; preds = %bb46
  call void @_ZdlPv(i8* %i48) #16
  br label %bb51

bb51:                                             ; preds = %bb50, %bb46
  %i52 = icmp eq i64 %i26, 0
  %i53 = add nsw i64 %i26, -1
  %i54 = load i8*, i8** %i15, align 8, !tbaa !58
  %i55 = icmp eq i8* %i54, %i17
  br i1 %i55, label %bb57, label %bb56

bb56:                                             ; preds = %bb51
  call void @_ZdlPv(i8* %i54) #16
  br label %bb57

bb57:                                             ; preds = %bb56, %bb51
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i12) #16
  br i1 %i52, label %bb68, label %bb25, !llvm.loop !70

bb68:                                             ; preds = %bb57, %bb
  ret void
}

; Function Attrs: inlinehint uwtable
define internal fastcc void @_ZSt10__pop_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_SF_RT0_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* %arg1, %"class.std::__cxx11::basic_string"* %arg2, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg3) unnamed_addr #10 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = alloca %"class.std::__cxx11::basic_string", align 8
  %i4 = alloca %"class.std::__cxx11::basic_string", align 8
  %i5 = bitcast %"class.std::__cxx11::basic_string"* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i5) #16
  %i6 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 2
  %i7 = bitcast %"class.std::__cxx11::basic_string"* %i to %union.anon**
  store %union.anon* %i6, %union.anon** %i7, align 8, !tbaa !67
  %i8 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i9 = load i8*, i8** %i8, align 8, !tbaa !58
  %i10 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 2
  %i11 = bitcast %union.anon* %i10 to i8*
  %i12 = icmp eq i8* %i9, %i11
  br i1 %i12, label %bb13, label %bb15

bb13:                                             ; preds = %bb
  %i14 = bitcast %union.anon* %i6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %i14, i8* nonnull align 8 dereferenceable(16) %i9, i64 16, i1 false) #16
  br label %bb20

bb15:                                             ; preds = %bb
  %i16 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 0, i32 0
  store i8* %i9, i8** %i16, align 8, !tbaa !58
  %i17 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 2, i32 0
  %i18 = load i64, i64* %i17, align 8, !tbaa !68
  %i19 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 2, i32 0
  store i64 %i18, i64* %i19, align 8, !tbaa !68
  br label %bb20

bb20:                                             ; preds = %bb15, %bb13
  %i21 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 1
  %i22 = load i64, i64* %i21, align 8, !tbaa !69
  %i23 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 1
  store i64 %i22, i64* %i23, align 8, !tbaa !69
  %i24 = bitcast %"class.std::__cxx11::basic_string"* %arg2 to %union.anon**
  store %union.anon* %i10, %union.anon** %i24, align 8, !tbaa !58
  store i64 0, i64* %i21, align 8, !tbaa !69
  store i8 0, i8* %i11, align 8, !tbaa !68
  %i25 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 0, i32 0, i32 0
  %i26 = load i8*, i8** %i25, align 8, !tbaa !58
  %i27 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 0, i32 2
  %i28 = bitcast %union.anon* %i27 to i8*
  %i29 = icmp eq i8* %i26, %i28
  br i1 %i29, label %bb30, label %bb40

bb30:                                             ; preds = %bb20
  %i31 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 0, i32 1
  %i32 = load i64, i64* %i31, align 8, !tbaa !69
  switch i64 %i32, label %bb35 [
    i64 0, label %bb36
    i64 1, label %bb33
  ]

bb33:                                             ; preds = %bb30
  %i34 = load i8, i8* %i26, align 1, !tbaa !68
  store i8 %i34, i8* %i11, align 1, !tbaa !68
  br label %bb36

bb35:                                             ; preds = %bb30
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 %i11, i8* align 1 %i26, i64 %i32, i1 false) #16
  br label %bb36

bb36:                                             ; preds = %bb35, %bb33, %bb30
  %i37 = load i64, i64* %i31, align 8, !tbaa !69
  store i64 %i37, i64* %i21, align 8, !tbaa !69
  %i38 = getelementptr inbounds i8, i8* %i11, i64 %i37
  store i8 0, i8* %i38, align 1, !tbaa !68
  %i39 = load i8*, i8** %i25, align 8, !tbaa !58
  br label %bb47

bb40:                                             ; preds = %bb20
  store i8* %i26, i8** %i8, align 8, !tbaa !58
  %i41 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 0, i32 1
  %i42 = load i64, i64* %i41, align 8, !tbaa !69
  store i64 %i42, i64* %i21, align 8, !tbaa !69
  %i43 = getelementptr %union.anon, %union.anon* %i27, i64 0, i32 0
  %i44 = load i64, i64* %i43, align 8, !tbaa !68
  %i45 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 2, i32 0
  store i64 %i44, i64* %i45, align 8, !tbaa !68
  %i46 = bitcast %"class.std::__cxx11::basic_string"* %arg to %union.anon**
  store %union.anon* %i27, %union.anon** %i46, align 8, !tbaa !58
  br label %bb47

bb47:                                             ; preds = %bb40, %bb36
  %i48 = phi i8* [ %i39, %bb36 ], [ %i28, %bb40 ]
  %i49 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 0, i32 1
  store i64 0, i64* %i49, align 8, !tbaa !69
  store i8 0, i8* %i48, align 1, !tbaa !68
  %i50 = ptrtoint %"class.std::__cxx11::basic_string"* %arg1 to i64
  %i51 = ptrtoint %"class.std::__cxx11::basic_string"* %arg to i64
  %i52 = sub i64 %i50, %i51
  %i53 = ashr exact i64 %i52, 5
  %i54 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 2
  %i55 = bitcast %"class.std::__cxx11::basic_string"* %i4 to %union.anon**
  store %union.anon* %i54, %union.anon** %i55, align 8, !tbaa !67
  %i56 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 0, i32 0
  %i57 = load i8*, i8** %i56, align 8, !tbaa !58
  %i58 = bitcast %union.anon* %i6 to i8*
  %i59 = icmp eq i8* %i57, %i58
  br i1 %i59, label %bb60, label %bb62

bb60:                                             ; preds = %bb47
  %i61 = bitcast %union.anon* %i54 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %i61, i8* nonnull align 8 dereferenceable(16) %i58, i64 16, i1 false) #16
  br label %bb67

bb62:                                             ; preds = %bb47
  %i63 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 0, i32 0
  store i8* %i57, i8** %i63, align 8, !tbaa !58
  %i64 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i, i64 0, i32 2, i32 0
  %i65 = load i64, i64* %i64, align 8, !tbaa !68
  %i66 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 2, i32 0
  store i64 %i65, i64* %i66, align 8, !tbaa !68
  br label %bb67

bb67:                                             ; preds = %bb62, %bb60
  %i68 = load i64, i64* %i23, align 8, !tbaa !69
  %i69 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 1
  store i64 %i68, i64* %i69, align 8, !tbaa !69
  store %union.anon* %i6, %union.anon** %i7, align 8, !tbaa !58
  store i64 0, i64* %i23, align 8, !tbaa !69
  store i8 0, i8* %i58, align 8, !tbaa !68
  call fastcc void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElS7_NS0_5__ops15_Iter_less_iterEEvT_T0_SG_T1_T2_(%"class.std::__cxx11::basic_string"* nonnull %arg, i64 0, i64 %i53, %"class.std::__cxx11::basic_string"* nonnull %i4)
  %i71 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 0, i32 0
  %i72 = load i8*, i8** %i71, align 8, !tbaa !58
  %i73 = bitcast %union.anon* %i54 to i8*
  %i74 = icmp eq i8* %i72, %i73
  br i1 %i74, label %bb76, label %bb75

bb75:                                             ; preds = %bb67
  call void @_ZdlPv(i8* %i72) #16
  br label %bb76

bb76:                                             ; preds = %bb75, %bb67
  %i77 = load i8*, i8** %i56, align 8, !tbaa !58
  %i78 = icmp eq i8* %i77, %i58
  br i1 %i78, label %bb80, label %bb79

bb79:                                             ; preds = %bb76
  call void @_ZdlPv(i8* %i77) #16
  br label %bb80

bb80:                                             ; preds = %bb79, %bb76
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i5) #16
  ret void
}

; Function Attrs: uwtable mustprogress
define internal fastcc void @_ZSt22__move_median_to_firstIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEEvT_SF_SF_SF_T0_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* %arg1, %"class.std::__cxx11::basic_string"* %arg2, %"class.std::__cxx11::basic_string"* %arg3) unnamed_addr #11 comdat personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 1
  %i4 = load i64, i64* %i, align 8, !tbaa !69
  %i5 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 1
  %i6 = load i64, i64* %i5, align 8, !tbaa !69
  %i7 = icmp ugt i64 %i4, %i6
  %i8 = select i1 %i7, i64 %i6, i64 %i4
  %i9 = icmp eq i64 %i8, 0
  br i1 %i9, label %bb17, label %bb10

bb10:                                             ; preds = %bb
  %i11 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i12 = load i8*, i8** %i11, align 8, !tbaa !58
  %i13 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 0, i32 0
  %i14 = load i8*, i8** %i13, align 8, !tbaa !58
  %i15 = tail call i32 @memcmp(i8* %i14, i8* %i12, i64 %i8) #16
  %i16 = icmp eq i32 %i15, 0
  br i1 %i16, label %bb17, label %bb24

bb17:                                             ; preds = %bb10, %bb
  %i18 = sub i64 %i4, %i6
  %i19 = icmp sgt i64 %i18, 2147483647
  br i1 %i19, label %bb72, label %bb20

bb20:                                             ; preds = %bb17
  %i21 = icmp sgt i64 %i18, -2147483648
  %i22 = select i1 %i21, i64 %i18, i64 -2147483648
  %i23 = trunc i64 %i22 to i32
  br label %bb24

bb24:                                             ; preds = %bb20, %bb10
  %i25 = phi i32 [ %i15, %bb10 ], [ %i23, %bb20 ]
  %i26 = icmp slt i32 %i25, 0
  br i1 %i26, label %bb27, label %bb72

bb27:                                             ; preds = %bb24
  %i28 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 1
  %i29 = load i64, i64* %i28, align 8, !tbaa !69
  %i30 = icmp ugt i64 %i6, %i29
  %i31 = select i1 %i30, i64 %i29, i64 %i6
  %i32 = icmp eq i64 %i31, 0
  br i1 %i32, label %bb40, label %bb33

bb33:                                             ; preds = %bb27
  %i34 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 0, i32 0
  %i35 = load i8*, i8** %i34, align 8, !tbaa !58
  %i36 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i37 = load i8*, i8** %i36, align 8, !tbaa !58
  %i38 = tail call i32 @memcmp(i8* %i37, i8* %i35, i64 %i31) #16
  %i39 = icmp eq i32 %i38, 0
  br i1 %i39, label %bb40, label %bb47

bb40:                                             ; preds = %bb33, %bb27
  %i41 = sub i64 %i6, %i29
  %i42 = icmp sgt i64 %i41, 2147483647
  br i1 %i42, label %bb50, label %bb43

bb43:                                             ; preds = %bb40
  %i44 = icmp sgt i64 %i41, -2147483648
  %i45 = select i1 %i44, i64 %i41, i64 -2147483648
  %i46 = trunc i64 %i45 to i32
  br label %bb47

bb47:                                             ; preds = %bb43, %bb33
  %i48 = phi i32 [ %i38, %bb33 ], [ %i46, %bb43 ]
  %i49 = icmp slt i32 %i48, 0
  br i1 %i49, label %bb117, label %bb50

bb50:                                             ; preds = %bb47, %bb40
  %i51 = icmp ugt i64 %i4, %i29
  %i52 = select i1 %i51, i64 %i29, i64 %i4
  %i53 = icmp eq i64 %i52, 0
  br i1 %i53, label %bb61, label %bb54

bb54:                                             ; preds = %bb50
  %i55 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 0, i32 0
  %i56 = load i8*, i8** %i55, align 8, !tbaa !58
  %i57 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 0, i32 0
  %i58 = load i8*, i8** %i57, align 8, !tbaa !58
  %i59 = tail call i32 @memcmp(i8* %i58, i8* %i56, i64 %i52) #16
  %i60 = icmp eq i32 %i59, 0
  br i1 %i60, label %bb61, label %bb68

bb61:                                             ; preds = %bb54, %bb50
  %i62 = sub i64 %i4, %i29
  %i63 = icmp sgt i64 %i62, 2147483647
  br i1 %i63, label %bb71, label %bb64

bb64:                                             ; preds = %bb61
  %i65 = icmp sgt i64 %i62, -2147483648
  %i66 = select i1 %i65, i64 %i62, i64 -2147483648
  %i67 = trunc i64 %i66 to i32
  br label %bb68

bb68:                                             ; preds = %bb64, %bb54
  %i69 = phi i32 [ %i59, %bb54 ], [ %i67, %bb64 ]
  %i70 = icmp slt i32 %i69, 0
  br i1 %i70, label %bb117, label %bb71

bb71:                                             ; preds = %bb68, %bb61
  br label %bb117

bb72:                                             ; preds = %bb24, %bb17
  %i73 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 1
  %i74 = load i64, i64* %i73, align 8, !tbaa !69
  %i75 = icmp ugt i64 %i4, %i74
  %i76 = select i1 %i75, i64 %i74, i64 %i4
  %i77 = icmp eq i64 %i76, 0
  br i1 %i77, label %bb85, label %bb78

bb78:                                             ; preds = %bb72
  %i79 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 0, i32 0
  %i80 = load i8*, i8** %i79, align 8, !tbaa !58
  %i81 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 0, i32 0
  %i82 = load i8*, i8** %i81, align 8, !tbaa !58
  %i83 = tail call i32 @memcmp(i8* %i82, i8* %i80, i64 %i76) #16
  %i84 = icmp eq i32 %i83, 0
  br i1 %i84, label %bb85, label %bb92

bb85:                                             ; preds = %bb78, %bb72
  %i86 = sub i64 %i4, %i74
  %i87 = icmp sgt i64 %i86, 2147483647
  br i1 %i87, label %bb95, label %bb88

bb88:                                             ; preds = %bb85
  %i89 = icmp sgt i64 %i86, -2147483648
  %i90 = select i1 %i89, i64 %i86, i64 -2147483648
  %i91 = trunc i64 %i90 to i32
  br label %bb92

bb92:                                             ; preds = %bb88, %bb78
  %i93 = phi i32 [ %i83, %bb78 ], [ %i91, %bb88 ]
  %i94 = icmp slt i32 %i93, 0
  br i1 %i94, label %bb117, label %bb95

bb95:                                             ; preds = %bb92, %bb85
  %i96 = icmp ugt i64 %i6, %i74
  %i97 = select i1 %i96, i64 %i74, i64 %i6
  %i98 = icmp eq i64 %i97, 0
  br i1 %i98, label %bb106, label %bb99

bb99:                                             ; preds = %bb95
  %i100 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 0, i32 0
  %i101 = load i8*, i8** %i100, align 8, !tbaa !58
  %i102 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i103 = load i8*, i8** %i102, align 8, !tbaa !58
  %i104 = tail call i32 @memcmp(i8* %i103, i8* %i101, i64 %i97) #16
  %i105 = icmp eq i32 %i104, 0
  br i1 %i105, label %bb106, label %bb113

bb106:                                            ; preds = %bb99, %bb95
  %i107 = sub i64 %i6, %i74
  %i108 = icmp sgt i64 %i107, 2147483647
  br i1 %i108, label %bb116, label %bb109

bb109:                                            ; preds = %bb106
  %i110 = icmp sgt i64 %i107, -2147483648
  %i111 = select i1 %i110, i64 %i107, i64 -2147483648
  %i112 = trunc i64 %i111 to i32
  br label %bb113

bb113:                                            ; preds = %bb109, %bb99
  %i114 = phi i32 [ %i104, %bb99 ], [ %i112, %bb109 ]
  %i115 = icmp slt i32 %i114, 0
  br i1 %i115, label %bb117, label %bb116

bb116:                                            ; preds = %bb113, %bb106
  br label %bb117

bb117:                                            ; preds = %bb116, %bb113, %bb92, %bb71, %bb68, %bb47
  %i118 = phi %"class.std::__cxx11::basic_string"* [ %arg2, %bb116 ], [ %arg1, %bb71 ], [ %arg2, %bb47 ], [ %arg3, %bb68 ], [ %arg1, %bb92 ], [ %arg3, %bb113 ]
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %arg, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %i118) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal fastcc %"class.std::__cxx11::basic_string"* @_ZSt21__unguarded_partitionIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEENS0_5__ops15_Iter_less_iterEET_SF_SF_SF_T0_(%"class.std::__cxx11::basic_string"* %arg, %"class.std::__cxx11::basic_string"* %arg1, %"class.std::__cxx11::basic_string"* %arg2) unnamed_addr #12 comdat personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 1
  %i3 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  br label %bb4

bb4:                                              ; preds = %bb61, %bb
  %i5 = phi %"class.std::__cxx11::basic_string"* [ %arg, %bb ], [ %i62, %bb61 ]
  %i6 = phi %"class.std::__cxx11::basic_string"* [ %arg1, %bb ], [ %i36, %bb61 ]
  %i7 = load i64, i64* %i, align 8, !tbaa !69
  br label %bb8

bb8:                                              ; preds = %bb32, %bb4
  %i9 = phi %"class.std::__cxx11::basic_string"* [ %i5, %bb4 ], [ %i33, %bb32 ]
  %i10 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i9, i64 0, i32 1
  %i11 = load i64, i64* %i10, align 8, !tbaa !69
  %i12 = icmp ugt i64 %i11, %i7
  %i13 = select i1 %i12, i64 %i7, i64 %i11
  %i14 = icmp eq i64 %i13, 0
  br i1 %i14, label %bb21, label %bb15

bb15:                                             ; preds = %bb8
  %i16 = load i8*, i8** %i3, align 8, !tbaa !58
  %i17 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i9, i64 0, i32 0, i32 0
  %i18 = load i8*, i8** %i17, align 8, !tbaa !58
  %i19 = tail call i32 @memcmp(i8* %i18, i8* %i16, i64 %i13) #16
  %i20 = icmp eq i32 %i19, 0
  br i1 %i20, label %bb21, label %bb28

bb21:                                             ; preds = %bb15, %bb8
  %i22 = sub i64 %i11, %i7
  %i23 = icmp sgt i64 %i22, 2147483647
  br i1 %i23, label %bb31, label %bb24

bb24:                                             ; preds = %bb21
  %i25 = icmp sgt i64 %i22, -2147483648
  %i26 = select i1 %i25, i64 %i22, i64 -2147483648
  %i27 = trunc i64 %i26 to i32
  br label %bb28

bb28:                                             ; preds = %bb24, %bb15
  %i29 = phi i32 [ %i19, %bb15 ], [ %i27, %bb24 ]
  %i30 = icmp slt i32 %i29, 0
  br i1 %i30, label %bb32, label %bb31

bb31:                                             ; preds = %bb28, %bb21
  br label %bb34

bb32:                                             ; preds = %bb28
  %i33 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i9, i64 1
  br label %bb8, !llvm.loop !71

bb34:                                             ; preds = %bb55, %bb31
  %i35 = phi %"class.std::__cxx11::basic_string"* [ %i36, %bb55 ], [ %i6, %bb31 ]
  %i36 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i35, i64 -1
  %i37 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i35, i64 -1, i32 1
  %i38 = load i64, i64* %i37, align 8, !tbaa !69
  %i39 = icmp ugt i64 %i7, %i38
  %i40 = select i1 %i39, i64 %i38, i64 %i7
  %i41 = icmp eq i64 %i40, 0
  br i1 %i41, label %bb48, label %bb42

bb42:                                             ; preds = %bb34
  %i43 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i36, i64 0, i32 0, i32 0
  %i44 = load i8*, i8** %i43, align 8, !tbaa !58
  %i45 = load i8*, i8** %i3, align 8, !tbaa !58
  %i46 = tail call i32 @memcmp(i8* %i45, i8* %i44, i64 %i40) #16
  %i47 = icmp eq i32 %i46, 0
  br i1 %i47, label %bb48, label %bb55

bb48:                                             ; preds = %bb42, %bb34
  %i49 = sub i64 %i7, %i38
  %i50 = icmp sgt i64 %i49, 2147483647
  br i1 %i50, label %bb58, label %bb51

bb51:                                             ; preds = %bb48
  %i52 = icmp sgt i64 %i49, -2147483648
  %i53 = select i1 %i52, i64 %i49, i64 -2147483648
  %i54 = trunc i64 %i53 to i32
  br label %bb55

bb55:                                             ; preds = %bb51, %bb42
  %i56 = phi i32 [ %i46, %bb42 ], [ %i54, %bb51 ]
  %i57 = icmp slt i32 %i56, 0
  br i1 %i57, label %bb34, label %bb58, !llvm.loop !72

bb58:                                             ; preds = %bb55, %bb48
  %i59 = icmp ult %"class.std::__cxx11::basic_string"* %i9, %i36
  br i1 %i59, label %bb61, label %bb60

bb60:                                             ; preds = %bb58
  ret %"class.std::__cxx11::basic_string"* %i9

bb61:                                             ; preds = %bb58
  tail call void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %i9, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %i36) #16
  %i62 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i9, i64 1
  br label %bb4, !llvm.loop !73
}

; Function Attrs: nounwind
declare void @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE4swapERS4_(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32), %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32)) local_unnamed_addr #7

; Function Attrs: uwtable
define internal fastcc void @_ZSt13__adjust_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElS7_NS0_5__ops15_Iter_less_iterEEvT_T0_SG_T1_T2_(%"class.std::__cxx11::basic_string"* %arg, i64 %arg1, i64 %arg2, %"class.std::__cxx11::basic_string"* %arg3) unnamed_addr #9 comdat personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = alloca %"class.std::ios_base::Init", align 1
  %i4 = alloca %"class.std::__cxx11::basic_string", align 8
  %i5 = add nsw i64 %arg2, -1
  %i6 = sdiv i64 %i5, 2
  %i7 = icmp sgt i64 %i6, %arg1
  br i1 %i7, label %bb8, label %bb88

bb8:                                              ; preds = %bb84, %bb
  %i9 = phi i64 [ %i39, %bb84 ], [ %arg1, %bb ]
  %i10 = shl i64 %i9, 1
  %i11 = add i64 %i10, 2
  %i12 = or i64 %i10, 1
  %i13 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i11, i32 1
  %i14 = load i64, i64* %i13, align 8, !tbaa !69
  %i15 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i12, i32 1
  %i16 = load i64, i64* %i15, align 8, !tbaa !69
  %i17 = icmp ugt i64 %i14, %i16
  %i18 = select i1 %i17, i64 %i16, i64 %i14
  %i19 = icmp eq i64 %i18, 0
  br i1 %i19, label %bb27, label %bb20

bb20:                                             ; preds = %bb8
  %i21 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i12, i32 0, i32 0
  %i22 = load i8*, i8** %i21, align 8, !tbaa !58
  %i23 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i11, i32 0, i32 0
  %i24 = load i8*, i8** %i23, align 8, !tbaa !58
  %i25 = tail call i32 @memcmp(i8* %i24, i8* %i22, i64 %i18) #16
  %i26 = icmp eq i32 %i25, 0
  br i1 %i26, label %bb27, label %bb34

bb27:                                             ; preds = %bb20, %bb8
  %i28 = sub i64 %i14, %i16
  %i29 = icmp sgt i64 %i28, 2147483647
  br i1 %i29, label %bb38, label %bb30

bb30:                                             ; preds = %bb27
  %i31 = icmp sgt i64 %i28, -2147483648
  %i32 = select i1 %i31, i64 %i28, i64 -2147483648
  %i33 = trunc i64 %i32 to i32
  br label %bb34

bb34:                                             ; preds = %bb30, %bb20
  %i35 = phi i32 [ %i25, %bb20 ], [ %i33, %bb30 ]
  %i36 = icmp slt i32 %i35, 0
  %i37 = select i1 %i36, i64 %i12, i64 %i11
  br label %bb38

bb38:                                             ; preds = %bb34, %bb27
  %i39 = phi i64 [ %i11, %bb27 ], [ %i37, %bb34 ]
  %i40 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i39
  %i41 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 0, i32 0
  %i42 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i40, i64 0, i32 0, i32 0
  %i43 = load i8*, i8** %i42, align 8, !tbaa !58
  %i44 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i39, i32 2
  %i45 = bitcast %union.anon* %i44 to i8*
  %i46 = icmp eq i8* %i43, %i45
  br i1 %i46, label %bb47, label %bb63

bb47:                                             ; preds = %bb38
  %i48 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i39, i32 1
  %i49 = load i64, i64* %i48, align 8, !tbaa !69
  %i50 = icmp eq i64 %i49, 0
  br i1 %i50, label %bb57, label %bb51

bb51:                                             ; preds = %bb47
  %i52 = load i8*, i8** %i41, align 8, !tbaa !58
  %i53 = icmp eq i64 %i49, 1
  br i1 %i53, label %bb54, label %bb56

bb54:                                             ; preds = %bb51
  %i55 = load i8, i8* %i43, align 1, !tbaa !68
  store i8 %i55, i8* %i52, align 1, !tbaa !68
  br label %bb57

bb56:                                             ; preds = %bb51
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %i52, i8* align 1 %i43, i64 %i49, i1 false) #16
  br label %bb57

bb57:                                             ; preds = %bb56, %bb54, %bb47
  %i58 = load i64, i64* %i48, align 8, !tbaa !69
  %i59 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 1
  store i64 %i58, i64* %i59, align 8, !tbaa !69
  %i60 = load i8*, i8** %i41, align 8, !tbaa !58
  %i61 = getelementptr inbounds i8, i8* %i60, i64 %i58
  store i8 0, i8* %i61, align 1, !tbaa !68
  %i62 = load i8*, i8** %i42, align 8, !tbaa !58
  br label %bb84

bb63:                                             ; preds = %bb38
  %i64 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 2
  %i65 = bitcast %union.anon* %i64 to i8*
  %i66 = load i8*, i8** %i41, align 8, !tbaa !58
  %i67 = icmp eq i8* %i66, %i65
  br i1 %i67, label %bb71, label %bb68

bb68:                                             ; preds = %bb63
  %i69 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 2, i32 0
  %i70 = load i64, i64* %i69, align 8, !tbaa !68
  br label %bb71

bb71:                                             ; preds = %bb68, %bb63
  %i72 = phi i64 [ undef, %bb63 ], [ %i70, %bb68 ]
  %i73 = phi i8* [ null, %bb63 ], [ %i66, %bb68 ]
  store i8* %i43, i8** %i41, align 8, !tbaa !58
  %i74 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i39, i32 1
  %i75 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 1
  %i76 = bitcast i64* %i74 to <2 x i64>*
  %i77 = load <2 x i64>, <2 x i64>* %i76, align 8, !tbaa !68
  %i78 = bitcast i64* %i75 to <2 x i64>*
  store <2 x i64> %i77, <2 x i64>* %i78, align 8, !tbaa !68
  %i79 = icmp eq i8* %i73, null
  br i1 %i79, label %bb82, label %bb80

bb80:                                             ; preds = %bb71
  store i8* %i73, i8** %i42, align 8, !tbaa !58
  %i81 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i39, i32 2, i32 0
  store i64 %i72, i64* %i81, align 8, !tbaa !68
  br label %bb84

bb82:                                             ; preds = %bb71
  %i83 = bitcast %"class.std::__cxx11::basic_string"* %i40 to %union.anon**
  store %union.anon* %i44, %union.anon** %i83, align 8, !tbaa !58
  br label %bb84

bb84:                                             ; preds = %bb82, %bb80, %bb57
  %i85 = phi i8* [ %i62, %bb57 ], [ %i73, %bb80 ], [ %i45, %bb82 ]
  %i86 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i39, i32 1
  store i64 0, i64* %i86, align 8, !tbaa !69
  store i8 0, i8* %i85, align 1, !tbaa !68
  %i87 = icmp slt i64 %i39, %i6
  br i1 %i87, label %bb8, label %bb88, !llvm.loop !74

bb88:                                             ; preds = %bb84, %bb
  %i89 = phi i64 [ %arg1, %bb ], [ %i39, %bb84 ]
  %i90 = and i64 %arg2, 1
  %i91 = icmp eq i64 %i90, 0
  br i1 %i91, label %bb92, label %bb146

bb92:                                             ; preds = %bb88
  %i93 = add nsw i64 %arg2, -2
  %i94 = sdiv i64 %i93, 2
  %i95 = icmp eq i64 %i89, %i94
  br i1 %i95, label %bb96, label %bb146

bb96:                                             ; preds = %bb92
  %i97 = shl i64 %i89, 1
  %i98 = or i64 %i97, 1
  %i99 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i98
  %i100 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i89, i32 0, i32 0
  %i101 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i99, i64 0, i32 0, i32 0
  %i102 = load i8*, i8** %i101, align 8, !tbaa !58
  %i103 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i98, i32 2
  %i104 = bitcast %union.anon* %i103 to i8*
  %i105 = icmp eq i8* %i102, %i104
  br i1 %i105, label %bb106, label %bb122

bb106:                                            ; preds = %bb96
  %i107 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i98, i32 1
  %i108 = load i64, i64* %i107, align 8, !tbaa !69
  %i109 = icmp eq i64 %i108, 0
  br i1 %i109, label %bb116, label %bb110

bb110:                                            ; preds = %bb106
  %i111 = load i8*, i8** %i100, align 8, !tbaa !58
  %i112 = icmp eq i64 %i108, 1
  br i1 %i112, label %bb113, label %bb115

bb113:                                            ; preds = %bb110
  %i114 = load i8, i8* %i102, align 1, !tbaa !68
  store i8 %i114, i8* %i111, align 1, !tbaa !68
  br label %bb116

bb115:                                            ; preds = %bb110
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %i111, i8* align 1 %i102, i64 %i108, i1 false) #16
  br label %bb116

bb116:                                            ; preds = %bb115, %bb113, %bb106
  %i117 = load i64, i64* %i107, align 8, !tbaa !69
  %i118 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i89, i32 1
  store i64 %i117, i64* %i118, align 8, !tbaa !69
  %i119 = load i8*, i8** %i100, align 8, !tbaa !58
  %i120 = getelementptr inbounds i8, i8* %i119, i64 %i117
  store i8 0, i8* %i120, align 1, !tbaa !68
  %i121 = load i8*, i8** %i101, align 8, !tbaa !58
  br label %bb143

bb122:                                            ; preds = %bb96
  %i123 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i89, i32 2
  %i124 = bitcast %union.anon* %i123 to i8*
  %i125 = load i8*, i8** %i100, align 8, !tbaa !58
  %i126 = icmp eq i8* %i125, %i124
  br i1 %i126, label %bb130, label %bb127

bb127:                                            ; preds = %bb122
  %i128 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i89, i32 2, i32 0
  %i129 = load i64, i64* %i128, align 8, !tbaa !68
  br label %bb130

bb130:                                            ; preds = %bb127, %bb122
  %i131 = phi i64 [ undef, %bb122 ], [ %i129, %bb127 ]
  %i132 = phi i8* [ null, %bb122 ], [ %i125, %bb127 ]
  store i8* %i102, i8** %i100, align 8, !tbaa !58
  %i133 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i98, i32 1
  %i134 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i89, i32 1
  %i135 = bitcast i64* %i133 to <2 x i64>*
  %i136 = load <2 x i64>, <2 x i64>* %i135, align 8, !tbaa !68
  %i137 = bitcast i64* %i134 to <2 x i64>*
  store <2 x i64> %i136, <2 x i64>* %i137, align 8, !tbaa !68
  %i138 = icmp eq i8* %i132, null
  br i1 %i138, label %bb141, label %bb139

bb139:                                            ; preds = %bb130
  store i8* %i132, i8** %i101, align 8, !tbaa !58
  %i140 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i98, i32 2, i32 0
  store i64 %i131, i64* %i140, align 8, !tbaa !68
  br label %bb143

bb141:                                            ; preds = %bb130
  %i142 = bitcast %"class.std::__cxx11::basic_string"* %i99 to %union.anon**
  store %union.anon* %i103, %union.anon** %i142, align 8, !tbaa !58
  br label %bb143

bb143:                                            ; preds = %bb141, %bb139, %bb116
  %i144 = phi i8* [ %i121, %bb116 ], [ %i132, %bb139 ], [ %i104, %bb141 ]
  %i145 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i98, i32 1
  store i64 0, i64* %i145, align 8, !tbaa !69
  store i8 0, i8* %i144, align 1, !tbaa !68
  br label %bb146

bb146:                                            ; preds = %bb143, %bb92, %bb88
  %i147 = phi i64 [ %i98, %bb143 ], [ %i89, %bb92 ], [ %i89, %bb88 ]
  %i148 = getelementptr inbounds %"class.std::ios_base::Init", %"class.std::ios_base::Init"* %i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %i148) #16
  %i149 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 2
  %i150 = bitcast %"class.std::__cxx11::basic_string"* %i4 to %union.anon**
  store %union.anon* %i149, %union.anon** %i150, align 8, !tbaa !67
  %i151 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 0, i32 0
  %i152 = load i8*, i8** %i151, align 8, !tbaa !58
  %i153 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 2
  %i154 = bitcast %union.anon* %i153 to i8*
  %i155 = icmp eq i8* %i152, %i154
  br i1 %i155, label %bb156, label %bb158

bb156:                                            ; preds = %bb146
  %i157 = bitcast %union.anon* %i149 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %i157, i8* nonnull align 8 dereferenceable(16) %i152, i64 16, i1 false) #16
  br label %bb163

bb158:                                            ; preds = %bb146
  %i159 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 0, i32 0
  store i8* %i152, i8** %i159, align 8, !tbaa !58
  %i160 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 2, i32 0
  %i161 = load i64, i64* %i160, align 8, !tbaa !68
  %i162 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 2, i32 0
  store i64 %i161, i64* %i162, align 8, !tbaa !68
  br label %bb163

bb163:                                            ; preds = %bb158, %bb156
  %i164 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 1
  %i165 = load i64, i64* %i164, align 8, !tbaa !69
  %i166 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 1
  store i64 %i165, i64* %i166, align 8, !tbaa !69
  %i167 = bitcast %"class.std::__cxx11::basic_string"* %arg3 to %union.anon**
  store %union.anon* %i153, %union.anon** %i167, align 8, !tbaa !58
  store i64 0, i64* %i164, align 8, !tbaa !69
  store i8 0, i8* %i154, align 8, !tbaa !68
  call fastcc void @_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElS7_NS0_5__ops14_Iter_less_valEEvT_T0_SG_T1_RT2_(%"class.std::__cxx11::basic_string"* %arg, i64 %i147, i64 %arg1, %"class.std::__cxx11::basic_string"* nonnull %i4, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %i)
  %i169 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i4, i64 0, i32 0, i32 0
  %i170 = load i8*, i8** %i169, align 8, !tbaa !58
  %i171 = bitcast %union.anon* %i149 to i8*
  %i172 = icmp eq i8* %i170, %i171
  br i1 %i172, label %bb174, label %bb173

bb173:                                            ; preds = %bb163
  call void @_ZdlPv(i8* %i170) #16
  br label %bb174

bb174:                                            ; preds = %bb173, %bb163
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %i148) #16
  ret void
}

; Function Attrs: uwtable
define internal fastcc void @_ZSt11__push_heapIN9__gnu_cxx17__normal_iteratorIPNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt6vectorIS7_SaIS7_EEEElS7_NS0_5__ops14_Iter_less_valEEvT_T0_SG_T1_RT2_(%"class.std::__cxx11::basic_string"* %arg, i64 %arg1, i64 %arg2, %"class.std::__cxx11::basic_string"* %arg3, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg4) unnamed_addr #9 comdat personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = icmp sgt i64 %arg1, %arg2
  br i1 %i, label %bb5, label %bb79

bb5:                                              ; preds = %bb
  %i6 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 1
  %i7 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 0, i32 0
  br label %bb8

bb8:                                              ; preds = %bb76, %bb5
  %i9 = phi i64 [ %arg1, %bb5 ], [ %i11, %bb76 ]
  %i10 = add nsw i64 %i9, -1
  %i11 = sdiv i64 %i10, 2
  %i12 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i11
  %i13 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i11, i32 1
  %i14 = load i64, i64* %i13, align 8, !tbaa !69
  %i15 = load i64, i64* %i6, align 8, !tbaa !69
  %i16 = icmp ugt i64 %i14, %i15
  %i17 = select i1 %i16, i64 %i15, i64 %i14
  %i18 = icmp eq i64 %i17, 0
  br i1 %i18, label %bb25, label %bb19

bb19:                                             ; preds = %bb8
  %i20 = load i8*, i8** %i7, align 8, !tbaa !58
  %i21 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i12, i64 0, i32 0, i32 0
  %i22 = load i8*, i8** %i21, align 8, !tbaa !58
  %i23 = tail call i32 @memcmp(i8* %i22, i8* %i20, i64 %i17) #16
  %i24 = icmp eq i32 %i23, 0
  br i1 %i24, label %bb25, label %bb32

bb25:                                             ; preds = %bb19, %bb8
  %i26 = sub i64 %i14, %i15
  %i27 = icmp sgt i64 %i26, 2147483647
  br i1 %i27, label %bb79, label %bb28

bb28:                                             ; preds = %bb25
  %i29 = icmp sgt i64 %i26, -2147483648
  %i30 = select i1 %i29, i64 %i26, i64 -2147483648
  %i31 = trunc i64 %i30 to i32
  br label %bb32

bb32:                                             ; preds = %bb28, %bb19
  %i33 = phi i32 [ %i23, %bb19 ], [ %i31, %bb28 ]
  %i34 = icmp slt i32 %i33, 0
  br i1 %i34, label %bb35, label %bb79

bb35:                                             ; preds = %bb32
  %i36 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 0, i32 0
  %i37 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i12, i64 0, i32 0, i32 0
  %i38 = load i8*, i8** %i37, align 8, !tbaa !58
  %i39 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i11, i32 2
  %i40 = bitcast %union.anon* %i39 to i8*
  %i41 = icmp eq i8* %i38, %i40
  br i1 %i41, label %bb42, label %bb56

bb42:                                             ; preds = %bb35
  %i43 = icmp eq i64 %i14, 0
  br i1 %i43, label %bb50, label %bb44

bb44:                                             ; preds = %bb42
  %i45 = load i8*, i8** %i36, align 8, !tbaa !58
  %i46 = icmp eq i64 %i14, 1
  br i1 %i46, label %bb47, label %bb49

bb47:                                             ; preds = %bb44
  %i48 = load i8, i8* %i38, align 1, !tbaa !68
  store i8 %i48, i8* %i45, align 1, !tbaa !68
  br label %bb50

bb49:                                             ; preds = %bb44
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %i45, i8* align 1 %i38, i64 %i14, i1 false) #16
  br label %bb50

bb50:                                             ; preds = %bb49, %bb47, %bb42
  %i51 = load i64, i64* %i13, align 8, !tbaa !69
  %i52 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 1
  store i64 %i51, i64* %i52, align 8, !tbaa !69
  %i53 = load i8*, i8** %i36, align 8, !tbaa !58
  %i54 = getelementptr inbounds i8, i8* %i53, i64 %i51
  store i8 0, i8* %i54, align 1, !tbaa !68
  %i55 = load i8*, i8** %i37, align 8, !tbaa !58
  br label %bb76

bb56:                                             ; preds = %bb35
  %i57 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 2
  %i58 = bitcast %union.anon* %i57 to i8*
  %i59 = load i8*, i8** %i36, align 8, !tbaa !58
  %i60 = icmp eq i8* %i59, %i58
  br i1 %i60, label %bb64, label %bb61

bb61:                                             ; preds = %bb56
  %i62 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 2, i32 0
  %i63 = load i64, i64* %i62, align 8, !tbaa !68
  br label %bb64

bb64:                                             ; preds = %bb61, %bb56
  %i65 = phi i64 [ undef, %bb56 ], [ %i63, %bb61 ]
  %i66 = phi i8* [ null, %bb56 ], [ %i59, %bb61 ]
  store i8* %i38, i8** %i36, align 8, !tbaa !58
  %i67 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 1
  store i64 %i14, i64* %i67, align 8, !tbaa !69
  %i68 = getelementptr %union.anon, %union.anon* %i39, i64 0, i32 0
  %i69 = load i64, i64* %i68, align 8, !tbaa !68
  %i70 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i9, i32 2, i32 0
  store i64 %i69, i64* %i70, align 8, !tbaa !68
  %i71 = icmp eq i8* %i66, null
  br i1 %i71, label %bb74, label %bb72

bb72:                                             ; preds = %bb64
  store i8* %i66, i8** %i37, align 8, !tbaa !58
  %i73 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i11, i32 2, i32 0
  store i64 %i65, i64* %i73, align 8, !tbaa !68
  br label %bb76

bb74:                                             ; preds = %bb64
  %i75 = bitcast %"class.std::__cxx11::basic_string"* %i12 to %union.anon**
  store %union.anon* %i39, %union.anon** %i75, align 8, !tbaa !58
  br label %bb76

bb76:                                             ; preds = %bb74, %bb72, %bb50
  %i77 = phi i8* [ %i55, %bb50 ], [ %i66, %bb72 ], [ %i40, %bb74 ]
  store i64 0, i64* %i13, align 8, !tbaa !69
  store i8 0, i8* %i77, align 1, !tbaa !68
  %i78 = icmp sgt i64 %i11, %arg2
  br i1 %i78, label %bb8, label %bb79, !llvm.loop !75

bb79:                                             ; preds = %bb76, %bb32, %bb25, %bb
  %i80 = phi i64 [ %arg1, %bb ], [ %i9, %bb25 ], [ %i11, %bb76 ], [ %i9, %bb32 ]
  %i81 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i80, i32 0, i32 0
  %i82 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 0, i32 0
  %i83 = load i8*, i8** %i82, align 8, !tbaa !58
  %i84 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 2
  %i85 = bitcast %union.anon* %i84 to i8*
  %i86 = icmp eq i8* %i83, %i85
  br i1 %i86, label %bb87, label %bb103

bb87:                                             ; preds = %bb79
  %i88 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 1
  %i89 = load i64, i64* %i88, align 8, !tbaa !69
  %i90 = icmp eq i64 %i89, 0
  br i1 %i90, label %bb97, label %bb91

bb91:                                             ; preds = %bb87
  %i92 = load i8*, i8** %i81, align 8, !tbaa !58
  %i93 = icmp eq i64 %i89, 1
  br i1 %i93, label %bb94, label %bb96

bb94:                                             ; preds = %bb91
  %i95 = load i8, i8* %i83, align 1, !tbaa !68
  store i8 %i95, i8* %i92, align 1, !tbaa !68
  br label %bb97

bb96:                                             ; preds = %bb91
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %i92, i8* align 1 %i83, i64 %i89, i1 false) #16
  br label %bb97

bb97:                                             ; preds = %bb96, %bb94, %bb87
  %i98 = load i64, i64* %i88, align 8, !tbaa !69
  %i99 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i80, i32 1
  store i64 %i98, i64* %i99, align 8, !tbaa !69
  %i100 = load i8*, i8** %i81, align 8, !tbaa !58
  %i101 = getelementptr inbounds i8, i8* %i100, i64 %i98
  store i8 0, i8* %i101, align 1, !tbaa !68
  %i102 = load i8*, i8** %i82, align 8, !tbaa !58
  br label %bb125

bb103:                                            ; preds = %bb79
  %i104 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i80, i32 2
  %i105 = bitcast %union.anon* %i104 to i8*
  %i106 = load i8*, i8** %i81, align 8, !tbaa !58
  %i107 = icmp eq i8* %i106, %i105
  br i1 %i107, label %bb111, label %bb108

bb108:                                            ; preds = %bb103
  %i109 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i80, i32 2, i32 0
  %i110 = load i64, i64* %i109, align 8, !tbaa !68
  br label %bb111

bb111:                                            ; preds = %bb108, %bb103
  %i112 = phi i64 [ undef, %bb103 ], [ %i110, %bb108 ]
  %i113 = phi i8* [ null, %bb103 ], [ %i106, %bb108 ]
  store i8* %i83, i8** %i81, align 8, !tbaa !58
  %i114 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 1
  %i115 = load i64, i64* %i114, align 8, !tbaa !69
  %i116 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i80, i32 1
  store i64 %i115, i64* %i116, align 8, !tbaa !69
  %i117 = getelementptr %union.anon, %union.anon* %i84, i64 0, i32 0
  %i118 = load i64, i64* %i117, align 8, !tbaa !68
  %i119 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg, i64 %i80, i32 2, i32 0
  store i64 %i118, i64* %i119, align 8, !tbaa !68
  %i120 = icmp eq i8* %i113, null
  br i1 %i120, label %bb123, label %bb121

bb121:                                            ; preds = %bb111
  store i8* %i113, i8** %i82, align 8, !tbaa !58
  %i122 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 2, i32 0
  store i64 %i112, i64* %i122, align 8, !tbaa !68
  br label %bb125

bb123:                                            ; preds = %bb111
  %i124 = bitcast %"class.std::__cxx11::basic_string"* %arg3 to %union.anon**
  store %union.anon* %i84, %union.anon** %i124, align 8, !tbaa !58
  br label %bb125

bb125:                                            ; preds = %bb123, %bb121, %bb97
  %i126 = phi i8* [ %i102, %bb97 ], [ %i113, %bb121 ], [ %i85, %bb123 ]
  %i127 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg3, i64 0, i32 1
  store i64 0, i64* %i127, align 8, !tbaa !69
  store i8 0, i8* %i126, align 1, !tbaa !68
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage", %"class.Kripke::Core::FieldStorage"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageIdEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage", %"class.Kripke::Core::FieldStorage"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load double**, double*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage", %"class.Kripke::Core::FieldStorage"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load double**, double*** %i3, align 8, !tbaa !78
  %i5 = icmp eq double** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load double**, double*** %i1, align 8, !tbaa !79
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi double** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq double** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast double** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage", %"class.Kripke::Core::FieldStorage"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage", %"class.Kripke::Core::FieldStorage"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage", %"class.Kripke::Core::FieldStorage"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage", %"class.Kripke::Core::FieldStorage"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi double** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load double*, double** %i39, align 8, !tbaa !78
  %i41 = icmp eq double* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast double* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds double*, double** %i39, i64 1
  %i46 = icmp eq double** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !88
}

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(i8*) local_unnamed_addr #2

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core9DomainVarD2Ev(%"class.Kripke::Core::DomainVar"* nonnull dereferenceable(88) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i1, align 8, !tbaa !85
  %i3 = icmp eq %"class.Kripke::SdomId"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"class.Kripke::SdomId"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i8 = load i64*, i64** %i7, align 8, !tbaa !82
  %i9 = icmp eq i64* %i8, null
  br i1 %i9, label %bb12, label %bb10

bb10:                                             ; preds = %bb6
  %i11 = bitcast i64* %i8 to i8*
  tail call void @_ZdlPv(i8* nonnull %i11) #16
  br label %bb12

bb12:                                             ; preds = %bb10, %bb6
  %i13 = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i14 = load i64*, i64** %i13, align 8, !tbaa !82
  %i15 = icmp eq i64* %i14, null
  br i1 %i15, label %bb18, label %bb16

bb16:                                             ; preds = %bb12
  %i17 = bitcast i64* %i14 to i8*
  tail call void @_ZdlPv(i8* nonnull %i17) #16
  br label %bb18

bb18:                                             ; preds = %bb16, %bb12
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core9DomainVarD0Ev(%"class.Kripke::Core::DomainVar"* nonnull dereferenceable(88) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i1, align 8, !tbaa !85
  %i3 = icmp eq %"class.Kripke::SdomId"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"class.Kripke::SdomId"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i8 = load i64*, i64** %i7, align 8, !tbaa !82
  %i9 = icmp eq i64* %i8, null
  br i1 %i9, label %bb12, label %bb10

bb10:                                             ; preds = %bb6
  %i11 = bitcast i64* %i8 to i8*
  tail call void @_ZdlPv(i8* nonnull %i11) #16
  br label %bb12

bb12:                                             ; preds = %bb10, %bb6
  %i13 = getelementptr inbounds %"class.Kripke::Core::DomainVar", %"class.Kripke::Core::DomainVar"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i14 = load i64*, i64** %i13, align 8, !tbaa !82
  %i15 = icmp eq i64* %i14, null
  br i1 %i15, label %bb18, label %bb16

bb16:                                             ; preds = %bb12
  %i17 = bitcast i64* %i14 to i8*
  tail call void @_ZdlPv(i8* nonnull %i17) #16
  br label %bb18

bb18:                                             ; preds = %bb16, %bb12
  %i19 = bitcast %"class.Kripke::Core::DomainVar"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i19) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageIdED0Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EED2Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !89
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EED0Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !89
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.33"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEED2Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !92
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEED0Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !92
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.33"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEED2Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !95
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEED0Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !95
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.33"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEED2Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !98
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEED0Ev(%"class.Kripke::Core::Field.33"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.61"*, %"struct.RAJA::TypedLayout.61"** %i1, align 8, !tbaa !98
  %i3 = icmp eq %"struct.RAJA::TypedLayout.61"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.61"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.33"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core3SetD2Ev(%"class.Kripke::Core::Set"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [6 x i8*] }, { [6 x i8*] }* @_ZTVN6Kripke4Core3SetE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i2 = load i64*, i64** %i1, align 8, !tbaa !82
  %i3 = icmp eq i64* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast i64* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i8 = load i64*, i64** %i7, align 8, !tbaa !82
  %i9 = icmp eq i64* %i8, null
  br i1 %i9, label %bb12, label %bb10

bb10:                                             ; preds = %bb6
  %i11 = bitcast i64* %i8 to i8*
  tail call void @_ZdlPv(i8* nonnull %i11) #16
  br label %bb12

bb12:                                             ; preds = %bb10, %bb6
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i13 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i14 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i13, align 8, !tbaa !85
  %i15 = icmp eq %"class.Kripke::SdomId"* %i14, null
  br i1 %i15, label %bb18, label %bb16

bb16:                                             ; preds = %bb12
  %i17 = bitcast %"class.Kripke::SdomId"* %i14 to i8*
  tail call void @_ZdlPv(i8* nonnull %i17) #16
  br label %bb18

bb18:                                             ; preds = %bb16, %bb12
  %i19 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i20 = load i64*, i64** %i19, align 8, !tbaa !82
  %i21 = icmp eq i64* %i20, null
  br i1 %i21, label %bb24, label %bb22

bb22:                                             ; preds = %bb18
  %i23 = bitcast i64* %i20 to i8*
  tail call void @_ZdlPv(i8* nonnull %i23) #16
  br label %bb24

bb24:                                             ; preds = %bb22, %bb18
  %i25 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i26 = load i64*, i64** %i25, align 8, !tbaa !82
  %i27 = icmp eq i64* %i26, null
  br i1 %i27, label %bb30, label %bb28

bb28:                                             ; preds = %bb24
  %i29 = bitcast i64* %i26 to i8*
  tail call void @_ZdlPv(i8* nonnull %i29) #16
  br label %bb30

bb30:                                             ; preds = %bb28, %bb24
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEED2Ev(%"class.Kripke::Core::Field"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout"*, %"struct.RAJA::TypedLayout"** %i1, align 8, !tbaa !101
  %i3 = icmp eq %"struct.RAJA::TypedLayout"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEED0Ev(%"class.Kripke::Core::Field"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout"*, %"struct.RAJA::TypedLayout"** %i1, align 8, !tbaa !101
  %i3 = icmp eq %"struct.RAJA::TypedLayout"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEED2Ev(%"class.Kripke::Core::Field"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout"*, %"struct.RAJA::TypedLayout"** %i1, align 8, !tbaa !104
  %i3 = icmp eq %"struct.RAJA::TypedLayout"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEED0Ev(%"class.Kripke::Core::Field"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout"*, %"struct.RAJA::TypedLayout"** %i1, align 8, !tbaa !104
  %i3 = icmp eq %"struct.RAJA::TypedLayout"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: uwtable
define internal fastcc void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPN12_GLOBAL__N_115QuadraturePointESt6vectorIS3_SaIS3_EEEElNS0_5__ops15_Iter_comp_iterIPFbRKS3_SC_EEEEvT_SG_T0_T1_(%"struct.(anonymous namespace)::QuadraturePoint"* %arg, %"struct.(anonymous namespace)::QuadraturePoint"* %arg1, i64 %arg2, i1 (%"struct.(anonymous namespace)::QuadraturePoint"*, %"struct.(anonymous namespace)::QuadraturePoint"*)* nocapture %arg3) unnamed_addr #9 {
bb:
  %i = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i4 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i5 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i6 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i7 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i8 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i9 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i10 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i11 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i12 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i13 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i14 = alloca %"struct.(anonymous namespace)::QuadraturePoint", align 8
  %i15 = ptrtoint %"struct.(anonymous namespace)::QuadraturePoint"* %arg1 to i64
  %i16 = ptrtoint %"struct.(anonymous namespace)::QuadraturePoint"* %arg to i64
  %i17 = sub i64 %i15, %i16
  %i18 = icmp sgt i64 %i17, 768
  br i1 %i18, label %bb19, label %bb266

bb19:                                             ; preds = %bb
  %i20 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 1
  %i21 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i7 to i8*
  %i22 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %arg to i8*
  %i23 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i6 to i8*
  %i24 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i4 to i8*
  %i25 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i20 to i8*
  %i26 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i to i8*
  %i27 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i5 to i8*
  %i28 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i8 to i8*
  %i29 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i9 to i8*
  br label %bb30

bb30:                                             ; preds = %bb262, %bb19
  %i31 = phi i64 [ %i17, %bb19 ], [ %i264, %bb262 ]
  %i32 = phi i64 [ %arg2, %bb19 ], [ %i208, %bb262 ]
  %i33 = phi %"struct.(anonymous namespace)::QuadraturePoint"* [ %arg1, %bb19 ], [ %i248, %bb262 ]
  %i34 = icmp eq i64 %i32, 0
  br i1 %i34, label %bb35, label %bb207

bb35:                                             ; preds = %bb30
  %i36 = udiv exact i64 %i31, 48
  %i37 = add nsw i64 %i36, -2
  %i38 = sdiv i64 %i37, 2
  %i39 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i14 to i8*
  %i40 = add nsw i64 %i36, -1
  %i41 = sdiv i64 %i40, 2
  %i42 = and i64 %i36, 1
  %i43 = icmp eq i64 %i42, 0
  %i44 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i13 to i8*
  br i1 %i43, label %bb45, label %bb97

bb45:                                             ; preds = %bb35
  %i46 = shl nuw nsw i64 %i38, 1
  %i47 = or i64 %i46, 1
  %i48 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i47
  %i49 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i38
  %i50 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i49 to i8*
  %i51 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i48 to i8*
  br label %bb52

bb52:                                             ; preds = %bb90, %bb45
  %i53 = phi i64 [ %i96, %bb90 ], [ %i38, %bb45 ]
  %i54 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i53
  %i55 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i54 to i8*
  %i56 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i56)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i39, i8* nonnull align 8 dereferenceable(48) %i55, i64 48, i1 false)
  %i57 = icmp sgt i64 %i41, %i53
  br i1 %i57, label %bb58, label %bb72

bb58:                                             ; preds = %bb58, %bb52
  %i59 = phi i64 [ %i66, %bb58 ], [ %i53, %bb52 ]
  %i60 = shl i64 %i59, 1
  %i61 = add i64 %i60, 2
  %i62 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i61
  %i63 = or i64 %i60, 1
  %i64 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i63
  %i65 = call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i62, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i64)
  %i66 = select i1 %i65, i64 %i63, i64 %i61
  %i67 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i66
  %i68 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i59
  %i69 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i68 to i8*
  %i70 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i67 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i69, i8* nonnull align 8 dereferenceable(48) %i70, i64 48, i1 false), !tbaa.struct !107
  %i71 = icmp slt i64 %i66, %i41
  br i1 %i71, label %bb58, label %bb72, !llvm.loop !112

bb72:                                             ; preds = %bb58, %bb52
  %i73 = phi i64 [ %i53, %bb52 ], [ %i66, %bb58 ]
  %i74 = icmp eq i64 %i73, %i38
  br i1 %i74, label %bb75, label %bb76

bb75:                                             ; preds = %bb72
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i50, i8* nonnull align 8 dereferenceable(48) %i51, i64 48, i1 false), !tbaa.struct !107
  br label %bb76

bb76:                                             ; preds = %bb75, %bb72
  %i77 = phi i64 [ %i47, %bb75 ], [ %i73, %bb72 ]
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i44)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i44, i8* nonnull align 8 dereferenceable(48) %i39, i64 48, i1 false)
  %i78 = icmp sgt i64 %i77, %i53
  br i1 %i78, label %bb79, label %bb90

bb79:                                             ; preds = %bb85, %bb76
  %i80 = phi i64 [ %i82, %bb85 ], [ %i77, %bb76 ]
  %i81 = add nsw i64 %i80, -1
  %i82 = sdiv i64 %i81, 2
  %i83 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i82
  %i84 = call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i83, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i13)
  br i1 %i84, label %bb85, label %bb90

bb85:                                             ; preds = %bb79
  %i86 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i80
  %i87 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i86 to i8*
  %i88 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i83 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i87, i8* nonnull align 8 dereferenceable(48) %i88, i64 48, i1 false), !tbaa.struct !107
  %i89 = icmp sgt i64 %i82, %i53
  br i1 %i89, label %bb79, label %bb90, !llvm.loop !113

bb90:                                             ; preds = %bb85, %bb79, %bb76
  %i91 = phi i64 [ %i77, %bb76 ], [ %i80, %bb79 ], [ %i82, %bb85 ]
  %i92 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i91
  %i93 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i92 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i93, i8* nonnull align 8 dereferenceable(48) %i44, i64 48, i1 false), !tbaa.struct !107
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i44)
  %i94 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i94)
  %i95 = icmp eq i64 %i53, 0
  %i96 = add nsw i64 %i53, -1
  br i1 %i95, label %bb139, label %bb52, !llvm.loop !114

bb97:                                             ; preds = %bb132, %bb35
  %i98 = phi i64 [ %i138, %bb132 ], [ %i38, %bb35 ]
  %i99 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i98
  %i100 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i99 to i8*
  %i101 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i101)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i39, i8* nonnull align 8 dereferenceable(48) %i100, i64 48, i1 false)
  %i102 = icmp sgt i64 %i41, %i98
  br i1 %i102, label %bb104, label %bb103

bb103:                                            ; preds = %bb97
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i44)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i44, i8* nonnull align 8 dereferenceable(48) %i39, i64 48, i1 false)
  br label %bb132

bb104:                                            ; preds = %bb104, %bb97
  %i105 = phi i64 [ %i112, %bb104 ], [ %i98, %bb97 ]
  %i106 = shl i64 %i105, 1
  %i107 = add i64 %i106, 2
  %i108 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i107
  %i109 = or i64 %i106, 1
  %i110 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i109
  %i111 = call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i108, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i110)
  %i112 = select i1 %i111, i64 %i109, i64 %i107
  %i113 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i112
  %i114 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i105
  %i115 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i114 to i8*
  %i116 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i113 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i115, i8* nonnull align 8 dereferenceable(48) %i116, i64 48, i1 false), !tbaa.struct !107
  %i117 = icmp slt i64 %i112, %i41
  br i1 %i117, label %bb104, label %bb118, !llvm.loop !112

bb118:                                            ; preds = %bb104
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i44)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i44, i8* nonnull align 8 dereferenceable(48) %i39, i64 48, i1 false)
  %i119 = icmp sgt i64 %i112, %i98
  br i1 %i119, label %bb120, label %bb132

bb120:                                            ; preds = %bb127, %bb118
  %i121 = phi i64 [ %i123, %bb127 ], [ %i112, %bb118 ]
  %i122 = add nsw i64 %i121, -1
  %i123 = sdiv i64 %i122, 2
  %i124 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i123
  %i125 = call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i124, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i13)
  br i1 %i125, label %bb127, label %bb132

bb127:                                            ; preds = %bb120
  %i128 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i121
  %i129 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i128 to i8*
  %i130 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i124 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i129, i8* nonnull align 8 dereferenceable(48) %i130, i64 48, i1 false), !tbaa.struct !107
  %i131 = icmp sgt i64 %i123, %i98
  br i1 %i131, label %bb120, label %bb132, !llvm.loop !113

bb132:                                            ; preds = %bb127, %bb120, %bb118, %bb103
  %i133 = phi i64 [ %i112, %bb118 ], [ %i98, %bb103 ], [ %i123, %bb127 ], [ %i121, %bb120 ]
  %i134 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i133
  %i135 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i134 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i135, i8* nonnull align 8 dereferenceable(48) %i44, i64 48, i1 false), !tbaa.struct !107
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i44)
  %i136 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i136)
  %i137 = icmp eq i64 %i98, 0
  %i138 = add nsw i64 %i98, -1
  br i1 %i137, label %bb139, label %bb97, !llvm.loop !114

bb139:                                            ; preds = %bb132, %bb90
  %i140 = icmp sgt i64 %i31, 48
  br i1 %i140, label %bb141, label %bb266

bb141:                                            ; preds = %bb139
  %i142 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i12 to i8*
  %i143 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i11 to i8*
  %i144 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i10 to i8*
  br label %bb145

bb145:                                            ; preds = %bb200, %bb141
  %i146 = phi %"struct.(anonymous namespace)::QuadraturePoint"* [ %i33, %bb141 ], [ %i147, %bb200 ]
  %i147 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %i146, i64 -1
  %i148 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i148)
  %i149 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i147 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i142, i8* nonnull align 8 dereferenceable(48) %i149, i64 48, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i149, i8* nonnull align 8 dereferenceable(48) %i22, i64 48, i1 false), !tbaa.struct !107
  %i150 = ptrtoint %"struct.(anonymous namespace)::QuadraturePoint"* %i147 to i64
  %i151 = sub i64 %i150, %i16
  %i152 = sdiv exact i64 %i151, 48
  %i153 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i153)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i143, i8* nonnull align 8 dereferenceable(48) %i142, i64 48, i1 false)
  %i154 = add nsw i64 %i152, -1
  %i155 = sdiv i64 %i154, 2
  %i156 = icmp sgt i64 %i151, 96
  br i1 %i156, label %bb157, label %bb171

bb157:                                            ; preds = %bb157, %bb145
  %i158 = phi i64 [ %i165, %bb157 ], [ 0, %bb145 ]
  %i159 = shl i64 %i158, 1
  %i160 = add i64 %i159, 2
  %i161 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i160
  %i162 = or i64 %i159, 1
  %i163 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i162
  %i164 = call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i161, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i163)
  %i165 = select i1 %i164, i64 %i162, i64 %i160
  %i166 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i165
  %i167 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i158
  %i168 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i167 to i8*
  %i169 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i166 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i168, i8* nonnull align 8 dereferenceable(48) %i169, i64 48, i1 false), !tbaa.struct !107
  %i170 = icmp slt i64 %i165, %i155
  br i1 %i170, label %bb157, label %bb171, !llvm.loop !112

bb171:                                            ; preds = %bb157, %bb145
  %i172 = phi i64 [ 0, %bb145 ], [ %i165, %bb157 ]
  %i173 = and i64 %i152, 1
  %i174 = icmp eq i64 %i173, 0
  br i1 %i174, label %bb175, label %bb186

bb175:                                            ; preds = %bb171
  %i176 = add nsw i64 %i152, -2
  %i177 = sdiv i64 %i176, 2
  %i178 = icmp eq i64 %i172, %i177
  br i1 %i178, label %bb179, label %bb186

bb179:                                            ; preds = %bb175
  %i180 = shl i64 %i172, 1
  %i181 = or i64 %i180, 1
  %i182 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i181
  %i183 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i172
  %i184 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i183 to i8*
  %i185 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i182 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i184, i8* nonnull align 8 dereferenceable(48) %i185, i64 48, i1 false), !tbaa.struct !107
  br label %bb186

bb186:                                            ; preds = %bb179, %bb175, %bb171
  %i187 = phi i64 [ %i181, %bb179 ], [ %i172, %bb175 ], [ %i172, %bb171 ]
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i144)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i144, i8* nonnull align 8 dereferenceable(48) %i143, i64 48, i1 false)
  %i188 = icmp sgt i64 %i187, 0
  br i1 %i188, label %bb189, label %bb200

bb189:                                            ; preds = %bb195, %bb186
  %i190 = phi i64 [ %i192, %bb195 ], [ %i187, %bb186 ]
  %i191 = add nsw i64 %i190, -1
  %i192 = sdiv i64 %i191, 2
  %i193 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i192
  %i194 = call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i193, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i10)
  br i1 %i194, label %bb195, label %bb200

bb195:                                            ; preds = %bb189
  %i196 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i190
  %i197 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i196 to i8*
  %i198 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i193 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i197, i8* nonnull align 8 dereferenceable(48) %i198, i64 48, i1 false), !tbaa.struct !107
  %i199 = icmp sgt i64 %i190, 2
  br i1 %i199, label %bb189, label %bb200, !llvm.loop !113

bb200:                                            ; preds = %bb195, %bb189, %bb186
  %i201 = phi i64 [ %i187, %bb186 ], [ %i190, %bb189 ], [ %i192, %bb195 ]
  %i202 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i201
  %i203 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i202 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i203, i8* nonnull align 8 dereferenceable(48) %i144, i64 48, i1 false), !tbaa.struct !107
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i144)
  %i204 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i204)
  %i205 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i12 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i205)
  %i206 = icmp sgt i64 %i151, 48
  br i1 %i206, label %bb145, label %bb266, !llvm.loop !115

bb207:                                            ; preds = %bb30
  %i208 = add nsw i64 %i32, -1
  %i209 = udiv i64 %i31, 96
  %i210 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %arg, i64 %i209
  %i211 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %i33, i64 -1
  %i212 = tail call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i20, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i210)
  br i1 %i212, label %bb213, label %bb228

bb213:                                            ; preds = %bb207
  %i214 = tail call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i210, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i211)
  br i1 %i214, label %bb215, label %bb219

bb215:                                            ; preds = %bb213
  %i216 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i216)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i28, i8* nonnull align 8 dereferenceable(48) %i22, i64 48, i1 false) #16, !tbaa.struct !107
  %i217 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i210 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i22, i8* nonnull align 8 dereferenceable(48) %i217, i64 48, i1 false) #16, !tbaa.struct !107
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i217, i8* nonnull align 8 dereferenceable(48) %i28, i64 48, i1 false) #16, !tbaa.struct !107
  %i218 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i218)
  br label %bb243

bb219:                                            ; preds = %bb213
  %i220 = tail call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i20, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i211)
  br i1 %i220, label %bb221, label %bb225

bb221:                                            ; preds = %bb219
  %i222 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i222)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i27, i8* nonnull align 8 dereferenceable(48) %i22, i64 48, i1 false) #16, !tbaa.struct !107
  %i223 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i211 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i22, i8* nonnull align 8 dereferenceable(48) %i223, i64 48, i1 false) #16, !tbaa.struct !107
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i223, i8* nonnull align 8 dereferenceable(48) %i27, i64 48, i1 false) #16, !tbaa.struct !107
  %i224 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i224)
  br label %bb243

bb225:                                            ; preds = %bb219
  %i226 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i226)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i26, i8* nonnull align 8 dereferenceable(48) %i22, i64 48, i1 false) #16, !tbaa.struct !107
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i22, i8* nonnull align 8 dereferenceable(48) %i25, i64 48, i1 false) #16, !tbaa.struct !107
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i25, i8* nonnull align 8 dereferenceable(48) %i26, i64 48, i1 false) #16, !tbaa.struct !107
  %i227 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i227)
  br label %bb243

bb228:                                            ; preds = %bb207
  %i229 = tail call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i20, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i211)
  br i1 %i229, label %bb230, label %bb233

bb230:                                            ; preds = %bb228
  %i231 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i231)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i24, i8* nonnull align 8 dereferenceable(48) %i22, i64 48, i1 false) #16, !tbaa.struct !107
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i22, i8* nonnull align 8 dereferenceable(48) %i25, i64 48, i1 false) #16, !tbaa.struct !107
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i25, i8* nonnull align 8 dereferenceable(48) %i24, i64 48, i1 false) #16, !tbaa.struct !107
  %i232 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i232)
  br label %bb243

bb233:                                            ; preds = %bb228
  %i234 = tail call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i210, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i211)
  br i1 %i234, label %bb235, label %bb239

bb235:                                            ; preds = %bb233
  %i236 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i236)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i23, i8* nonnull align 8 dereferenceable(48) %i22, i64 48, i1 false) #16, !tbaa.struct !107
  %i237 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i211 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i22, i8* nonnull align 8 dereferenceable(48) %i237, i64 48, i1 false) #16, !tbaa.struct !107
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i237, i8* nonnull align 8 dereferenceable(48) %i23, i64 48, i1 false) #16, !tbaa.struct !107
  %i238 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i238)
  br label %bb243

bb239:                                            ; preds = %bb233
  %i240 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i240)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i21, i8* nonnull align 8 dereferenceable(48) %i22, i64 48, i1 false) #16, !tbaa.struct !107
  %i241 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i210 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i22, i8* nonnull align 8 dereferenceable(48) %i241, i64 48, i1 false) #16, !tbaa.struct !107
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i241, i8* nonnull align 8 dereferenceable(48) %i21, i64 48, i1 false) #16, !tbaa.struct !107
  %i242 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i242)
  br label %bb243

bb243:                                            ; preds = %bb239, %bb235, %bb230, %bb225, %bb221, %bb215
  br label %bb244

bb244:                                            ; preds = %bb257, %bb243
  %i245 = phi %"struct.(anonymous namespace)::QuadraturePoint"* [ %i253, %bb257 ], [ %i33, %bb243 ]
  %i246 = phi %"struct.(anonymous namespace)::QuadraturePoint"* [ %i250, %bb257 ], [ %i20, %bb243 ]
  br label %bb247

bb247:                                            ; preds = %bb247, %bb244
  %i248 = phi %"struct.(anonymous namespace)::QuadraturePoint"* [ %i246, %bb244 ], [ %i250, %bb247 ]
  %i249 = tail call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i248, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %arg)
  %i250 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %i248, i64 1
  br i1 %i249, label %bb247, label %bb251, !llvm.loop !116

bb251:                                            ; preds = %bb251, %bb247
  %i252 = phi %"struct.(anonymous namespace)::QuadraturePoint"* [ %i253, %bb251 ], [ %i245, %bb247 ]
  %i253 = getelementptr inbounds %"struct.(anonymous namespace)::QuadraturePoint", %"struct.(anonymous namespace)::QuadraturePoint"* %i252, i64 -1
  %i254 = tail call zeroext i1 %arg3(%"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %arg, %"struct.(anonymous namespace)::QuadraturePoint"* nonnull align 8 dereferenceable(48) %i253)
  br i1 %i254, label %bb251, label %bb255, !llvm.loop !117

bb255:                                            ; preds = %bb251
  %i256 = icmp ult %"struct.(anonymous namespace)::QuadraturePoint"* %i248, %i253
  br i1 %i256, label %bb257, label %bb262

bb257:                                            ; preds = %bb255
  %i258 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %i258)
  %i259 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i248 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i29, i8* nonnull align 8 dereferenceable(48) %i259, i64 48, i1 false) #16, !tbaa.struct !107
  %i260 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i253 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i259, i8* nonnull align 8 dereferenceable(48) %i260, i64 48, i1 false) #16, !tbaa.struct !107
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i260, i8* nonnull align 8 dereferenceable(48) %i29, i64 48, i1 false) #16, !tbaa.struct !107
  %i261 = bitcast %"struct.(anonymous namespace)::QuadraturePoint"* %i9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %i261)
  br label %bb244, !llvm.loop !118

bb262:                                            ; preds = %bb255
  tail call fastcc void @_ZSt16__introsort_loopIN9__gnu_cxx17__normal_iteratorIPN12_GLOBAL__N_115QuadraturePointESt6vectorIS3_SaIS3_EEEElNS0_5__ops15_Iter_comp_iterIPFbRKS3_SC_EEEEvT_SG_T0_T1_(%"struct.(anonymous namespace)::QuadraturePoint"* %i248, %"struct.(anonymous namespace)::QuadraturePoint"* %i33, i64 %i208, i1 (%"struct.(anonymous namespace)::QuadraturePoint"*, %"struct.(anonymous namespace)::QuadraturePoint"*)* %arg3)
  %i263 = ptrtoint %"struct.(anonymous namespace)::QuadraturePoint"* %i248 to i64
  %i264 = sub i64 %i263, %i16
  %i265 = icmp sgt i64 %i264, 768
  br i1 %i265, label %bb30, label %bb266, !llvm.loop !119

bb266:                                            ; preds = %bb262, %bb200, %bb139, %bb
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i3, align 8, !tbaa !78
  %i5 = icmp eq %"class.Kripke::SdomId"** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !120
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi %"class.Kripke::SdomId"** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq %"class.Kripke::SdomId"** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast %"class.Kripke::SdomId"** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi %"class.Kripke::SdomId"** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, align 8, !tbaa !78
  %i41 = icmp eq %"class.Kripke::SdomId"* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast %"class.Kripke::SdomId"* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, i64 1
  %i46 = icmp eq %"class.Kripke::SdomId"** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !123
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED0Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.242"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEED2Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !124
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEED0Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !124
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.246"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEED2Ev(%"class.Kripke::Core::Field.61"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"** %i1, align 8, !tbaa !127
  %i3 = icmp eq %"struct.RAJA::TypedLayout.67"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.67"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEED0Ev(%"class.Kripke::Core::Field.61"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"** %i1, align 8, !tbaa !127
  %i3 = icmp eq %"struct.RAJA::TypedLayout.67"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.67"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.61"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEED2Ev(%"class.Kripke::Core::Field.61"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"** %i1, align 8, !tbaa !130
  %i3 = icmp eq %"struct.RAJA::TypedLayout.67"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.67"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEED0Ev(%"class.Kripke::Core::Field.61"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"** %i1, align 8, !tbaa !130
  %i3 = icmp eq %"struct.RAJA::TypedLayout.67"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.67"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.61"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageIiED2Ev(%"class.Kripke::Core::FieldStorage.49"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49", %"class.Kripke::Core::FieldStorage.49"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageIiEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49", %"class.Kripke::Core::FieldStorage.49"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load i32**, i32*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49", %"class.Kripke::Core::FieldStorage.49"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load i32**, i32*** %i3, align 8, !tbaa !78
  %i5 = icmp eq i32** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load i32**, i32*** %i1, align 8, !tbaa !133
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi i32** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq i32** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast i32** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49", %"class.Kripke::Core::FieldStorage.49"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49", %"class.Kripke::Core::FieldStorage.49"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49", %"class.Kripke::Core::FieldStorage.49"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49", %"class.Kripke::Core::FieldStorage.49"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi i32** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load i32*, i32** %i39, align 8, !tbaa !78
  %i41 = icmp eq i32* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast i32* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds i32*, i32** %i39, i64 1
  %i46 = icmp eq i32** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !136
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageIiED0Ev(%"class.Kripke::Core::FieldStorage.49"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageIiED2Ev(%"class.Kripke::Core::FieldStorage.49"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.49"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIiJNS_9DirectionEEED2Ev(%"class.Kripke::Core::Field.48.258"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIiJNS_9DirectionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !137
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIiED2Ev(%"class.Kripke::Core::FieldStorage.49"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIiJNS_9DirectionEEED0Ev(%"class.Kripke::Core::Field.48.258"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIiJNS_9DirectionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !137
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIiED2Ev(%"class.Kripke::Core::FieldStorage.49"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.48.258"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionEEED2Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !137
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_9DirectionEEED0Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_9DirectionEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !137
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.35"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_8LegendreEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageINS_8LegendreEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i3, align 8, !tbaa !78
  %i5 = icmp eq %"class.Kripke::SdomId"** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !140
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi %"class.Kripke::SdomId"** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq %"class.Kripke::SdomId"** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast %"class.Kripke::SdomId"** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi %"class.Kripke::SdomId"** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, align 8, !tbaa !78
  %i41 = icmp eq %"class.Kripke::SdomId"* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast %"class.Kripke::SdomId"* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, i64 1
  %i46 = icmp eq %"class.Kripke::SdomId"** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !143
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_8LegendreEED0Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageINS_8LegendreEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.242"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEED2Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !144
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_8LegendreEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEED0Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !144
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_8LegendreEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.246"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEED2Ev(%"class.Kripke::Core::Field.61"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"** %i1, align 8, !tbaa !147
  %i3 = icmp eq %"struct.RAJA::TypedLayout.67"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.67"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEED0Ev(%"class.Kripke::Core::Field.61"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.67"*, %"struct.RAJA::TypedLayout.67"** %i1, align 8, !tbaa !147
  %i3 = icmp eq %"struct.RAJA::TypedLayout.67"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.67"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.61", %"class.Kripke::Core::Field.61"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.61"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_7MixElemEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageINS_7MixElemEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i3, align 8, !tbaa !78
  %i5 = icmp eq %"class.Kripke::SdomId"** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !150
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi %"class.Kripke::SdomId"** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq %"class.Kripke::SdomId"** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast %"class.Kripke::SdomId"** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi %"class.Kripke::SdomId"** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, align 8, !tbaa !78
  %i41 = icmp eq %"class.Kripke::SdomId"* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast %"class.Kripke::SdomId"* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, i64 1
  %i46 = icmp eq %"class.Kripke::SdomId"** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !153
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_7MixElemEED0Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageINS_7MixElemEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.242"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEED2Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !154
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_7MixElemEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEED0Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !154
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_7MixElemEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.246"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIiJNS_4ZoneEEED2Ev(%"class.Kripke::Core::Field.48.258"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIiJNS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !154
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIiED2Ev(%"class.Kripke::Core::FieldStorage.49"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIiJNS_4ZoneEEED0Ev(%"class.Kripke::Core::Field.48.258"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIiJNS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !154
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIiED2Ev(%"class.Kripke::Core::FieldStorage.49"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.48.258"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_7MixElemEEED2Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_7MixElemEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !157
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_7MixElemEEED0Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_7MixElemEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !157
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.35"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_8MaterialEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageINS_8MaterialEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i3, align 8, !tbaa !78
  %i5 = icmp eq %"class.Kripke::SdomId"** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !160
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi %"class.Kripke::SdomId"** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq %"class.Kripke::SdomId"** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast %"class.Kripke::SdomId"** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi %"class.Kripke::SdomId"** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, align 8, !tbaa !78
  %i41 = icmp eq %"class.Kripke::SdomId"* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast %"class.Kripke::SdomId"* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, i64 1
  %i46 = icmp eq %"class.Kripke::SdomId"** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !163
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_8MaterialEED0Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageINS_8MaterialEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.242"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEED2Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !157
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_8MaterialEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEED0Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !157
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_8MaterialEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.246"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_4ZoneEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageINS_4ZoneEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i3, align 8, !tbaa !78
  %i5 = icmp eq %"class.Kripke::SdomId"** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !164
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi %"class.Kripke::SdomId"** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq %"class.Kripke::SdomId"** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast %"class.Kripke::SdomId"** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi %"class.Kripke::SdomId"** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, align 8, !tbaa !78
  %i41 = icmp eq %"class.Kripke::SdomId"* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast %"class.Kripke::SdomId"* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, i64 1
  %i46 = icmp eq %"class.Kripke::SdomId"** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !167
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_4ZoneEED0Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageINS_4ZoneEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.242"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEED2Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !157
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_4ZoneEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEED0Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !157
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_4ZoneEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.246"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_4ZoneEEED2Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !154
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_4ZoneEEED0Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_4ZoneEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !154
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.35"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5ZoneKEEED2Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5ZoneKEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !168
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5ZoneKEEED0Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5ZoneKEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !168
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.35"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5ZoneJEEED2Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5ZoneJEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !171
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5ZoneJEEED0Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5ZoneJEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !171
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.35"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5ZoneIEEED2Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5ZoneIEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !174
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIdJNS_5ZoneIEEED0Ev(%"class.Kripke::Core::Field.35"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIdJNS_5ZoneIEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !174
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.35", %"class.Kripke::Core::Field.35"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIdED2Ev(%"class.Kripke::Core::FieldStorage"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.35"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageIlED2Ev(%"class.Kripke::Core::FieldStorage.49.396"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49.396", %"class.Kripke::Core::FieldStorage.49.396"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageIlEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49.396", %"class.Kripke::Core::FieldStorage.49.396"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load i64**, i64*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49.396", %"class.Kripke::Core::FieldStorage.49.396"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load i64**, i64*** %i3, align 8, !tbaa !78
  %i5 = icmp eq i64** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load i64**, i64*** %i1, align 8, !tbaa !177
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi i64** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq i64** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast i64** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49.396", %"class.Kripke::Core::FieldStorage.49.396"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49.396", %"class.Kripke::Core::FieldStorage.49.396"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49.396", %"class.Kripke::Core::FieldStorage.49.396"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.49.396", %"class.Kripke::Core::FieldStorage.49.396"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi i64** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load i64*, i64** %i39, align 8, !tbaa !78
  %i41 = icmp eq i64* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast i64* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds i64*, i64** %i39, i64 1
  %i46 = icmp eq i64** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !180
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageIlED0Ev(%"class.Kripke::Core::FieldStorage.49.396"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageIlED2Ev(%"class.Kripke::Core::FieldStorage.49.396"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.49.396"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEED2Ev(%"class.Kripke::Core::Field.48.397"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.48.397", %"class.Kripke::Core::Field.48.397"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.48.397", %"class.Kripke::Core::Field.48.397"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !181
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.48.397", %"class.Kripke::Core::Field.48.397"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIlED2Ev(%"class.Kripke::Core::FieldStorage.49.396"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEED0Ev(%"class.Kripke::Core::Field.48.397"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.48.397", %"class.Kripke::Core::Field.48.397"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.48.397", %"class.Kripke::Core::Field.48.397"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !181
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.48.397", %"class.Kripke::Core::Field.48.397"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageIlED2Ev(%"class.Kripke::Core::FieldStorage.49.396"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.48.397"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core12FieldStorageINS_6SdomIdEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 0
  %i2 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !78
  %i3 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 3, i32 0, i32 0, i32 1
  %i4 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i3, align 8, !tbaa !78
  %i5 = icmp eq %"class.Kripke::SdomId"** %i2, %i4
  br i1 %i5, label %bb8, label %bb38

bb6:                                              ; preds = %bb44
  %i7 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i1, align 8, !tbaa !184
  br label %bb8

bb8:                                              ; preds = %bb6, %bb
  %i9 = phi %"class.Kripke::SdomId"** [ %i7, %bb6 ], [ %i2, %bb ]
  %i10 = icmp eq %"class.Kripke::SdomId"** %i9, null
  br i1 %i10, label %bb13, label %bb11

bb11:                                             ; preds = %bb8
  %i12 = bitcast %"class.Kripke::SdomId"** %i9 to i8*
  tail call void @_ZdlPv(i8* nonnull %i12) #16
  br label %bb13

bb13:                                             ; preds = %bb11, %bb8
  %i14 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 2, i32 0, i32 0, i32 0
  %i15 = load i64*, i64** %i14, align 8, !tbaa !82
  %i16 = icmp eq i64* %i15, null
  br i1 %i16, label %bb19, label %bb17

bb17:                                             ; preds = %bb13
  %i18 = bitcast i64* %i15 to i8*
  tail call void @_ZdlPv(i8* nonnull %i18) #16
  br label %bb19

bb19:                                             ; preds = %bb17, %bb13
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core9DomainVarE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i20 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i21 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i20, align 8, !tbaa !85
  %i22 = icmp eq %"class.Kripke::SdomId"* %i21, null
  br i1 %i22, label %bb25, label %bb23

bb23:                                             ; preds = %bb19
  %i24 = bitcast %"class.Kripke::SdomId"* %i21 to i8*
  tail call void @_ZdlPv(i8* nonnull %i24) #16
  br label %bb25

bb25:                                             ; preds = %bb23, %bb19
  %i26 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 2, i32 0, i32 0, i32 0
  %i27 = load i64*, i64** %i26, align 8, !tbaa !82
  %i28 = icmp eq i64* %i27, null
  br i1 %i28, label %bb31, label %bb29

bb29:                                             ; preds = %bb25
  %i30 = bitcast i64* %i27 to i8*
  tail call void @_ZdlPv(i8* nonnull %i30) #16
  br label %bb31

bb31:                                             ; preds = %bb29, %bb25
  %i32 = getelementptr inbounds %"class.Kripke::Core::FieldStorage.242", %"class.Kripke::Core::FieldStorage.242"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i33 = load i64*, i64** %i32, align 8, !tbaa !82
  %i34 = icmp eq i64* %i33, null
  br i1 %i34, label %bb37, label %bb35

bb35:                                             ; preds = %bb31
  %i36 = bitcast i64* %i33 to i8*
  tail call void @_ZdlPv(i8* nonnull %i36) #16
  br label %bb37

bb37:                                             ; preds = %bb35, %bb31
  ret void

bb38:                                             ; preds = %bb44, %bb
  %i39 = phi %"class.Kripke::SdomId"** [ %i45, %bb44 ], [ %i2, %bb ]
  %i40 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, align 8, !tbaa !78
  %i41 = icmp eq %"class.Kripke::SdomId"* %i40, null
  br i1 %i41, label %bb44, label %bb42

bb42:                                             ; preds = %bb38
  %i43 = bitcast %"class.Kripke::SdomId"* %i40 to i8*
  tail call void @_ZdaPv(i8* %i43) #17
  br label %bb44

bb44:                                             ; preds = %bb42, %bb38
  %i45 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i39, i64 1
  %i46 = icmp eq %"class.Kripke::SdomId"** %i45, %i4
  br i1 %i46, label %bb6, label %bb38, !llvm.loop !187
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED0Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %arg) #16
  %i = bitcast %"class.Kripke::Core::FieldStorage.242"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEED2Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !181
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEED0Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !181
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_6SdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.246"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEED2Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !188
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEED0Ev(%"class.Kripke::Core::Field.246"* nonnull dereferenceable(168) %arg) unnamed_addr #12 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  %i1 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i2 = load %"struct.RAJA::TypedLayout.47.255"*, %"struct.RAJA::TypedLayout.47.255"** %i1, align 8, !tbaa !188
  %i3 = icmp eq %"struct.RAJA::TypedLayout.47.255"* %i2, null
  br i1 %i3, label %bb6, label %bb4

bb4:                                              ; preds = %bb
  %i5 = bitcast %"struct.RAJA::TypedLayout.47.255"* %i2 to i8*
  tail call void @_ZdlPv(i8* nonnull %i5) #16
  br label %bb6

bb6:                                              ; preds = %bb4, %bb
  %i7 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg, i64 0, i32 0
  tail call void @_ZN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEED2Ev(%"class.Kripke::Core::FieldStorage.242"* nonnull dereferenceable(144) %i7) #16
  %i8 = bitcast %"class.Kripke::Core::Field.246"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i8) #17
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke4Core3SetD0Ev(%"class.Kripke::Core::Set"* nonnull dereferenceable(144) %arg) unnamed_addr #12 comdat align 2 {
bb:
  tail call void @llvm.trap() #18
  unreachable
}

; Function Attrs: cold noreturn nounwind
declare void @llvm.trap() #13

; Function Attrs: norecurse nounwind readonly uwtable willreturn mustprogress
define internal i64 @_ZNK6Kripke4Core3Set7dimSizeENS_6SdomIdEm(%"class.Kripke::Core::Set"* nocapture nonnull readonly dereferenceable(144) %arg, i64 %arg1, i64 %arg2) unnamed_addr #14 align 2 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i3 = load i64*, i64** %i, align 8, !tbaa !82
  %i4 = getelementptr inbounds i64, i64* %i3, i64 %arg1
  %i5 = load i64, i64* %i4, align 8, !tbaa !191
  %i6 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg, i64 0, i32 1, i32 0, i32 0, i32 0
  %i7 = load i64*, i64** %i6, align 8, !tbaa !82
  %i8 = getelementptr inbounds i64, i64* %i7, i64 %i5
  %i9 = load i64, i64* %i8, align 8, !tbaa !191
  ret i64 %i9
}

declare void @__cxa_pure_virtual() unnamed_addr

; Function Attrs: uwtable
define internal fastcc void @_ZN6Kripke6Kernel10scatteringERNS_4Core9DataStoreE(%"class.Kripke::Core::DataStore"* nonnull readonly align 8 dereferenceable(48) %arg) unnamed_addr #9 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = alloca i64, align 8
  %i1 = alloca %class.anon.509, align 8
  %i4 = alloca i64, align 8
  %i7 = alloca i64, align 8
  %i10 = alloca i64, align 8
  %i13 = alloca i64, align 8
  %i34 = alloca %"class.Kripke::BlockTimer", align 8
  %i35 = alloca %"class.std::__cxx11::basic_string", align 8
  %i36 = alloca %"class.std::__cxx11::basic_string", align 8
  %i37 = alloca %"class.std::__cxx11::basic_string", align 8
  %i38 = alloca %"class.std::__cxx11::basic_string", align 8
  %i39 = alloca %"class.std::__cxx11::basic_string", align 8
  %i40 = alloca %"class.std::__cxx11::basic_string", align 8
  %i41 = alloca %"class.std::__cxx11::basic_string", align 8
  %i42 = alloca %"class.std::__cxx11::basic_string", align 8
  %i43 = alloca %"class.std::__cxx11::basic_string", align 8
  %i44 = alloca %"class.std::__cxx11::basic_string", align 8
  %i45 = alloca %"class.std::__cxx11::basic_string", align 8
  %i46 = alloca %"class.std::__cxx11::basic_string", align 8
  %i47 = alloca %"class.std::__cxx11::basic_string", align 8
  %i48 = alloca %"class.std::__cxx11::basic_string", align 8
  %i49 = alloca %"class.std::__cxx11::basic_string", align 8
  %i50 = alloca %"class.Kripke::SdomId", align 8
  %i51 = alloca %"class.Kripke::SdomId", align 8
  %i52 = alloca %"class.std::ios_base::Init", align 1
  %i53 = bitcast %"class.Kripke::BlockTimer"* %i34 to i8*
  call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %i53) #16
  %i54 = bitcast %"class.std::__cxx11::basic_string"* %i35 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i54) #16
  %i55 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i35, i64 0, i32 2
  %i56 = bitcast %"class.std::__cxx11::basic_string"* %i35 to %union.anon**
  store %union.anon* %i55, %union.anon** %i56, align 8, !tbaa !67
  %i57 = bitcast %union.anon* %i55 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(6) %i57, i8* nonnull align 1 dereferenceable(6) getelementptr inbounds ([7 x i8], [7 x i8]* @.str.3.259, i64 0, i64 0), i64 6, i1 false) #16
  %i59 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i35, i64 0, i32 1
  store i64 6, i64* %i59, align 8, !tbaa !69
  %i60 = getelementptr inbounds i8, i8* %i57, i64 6
  store i8 0, i8* %i60, align 2, !tbaa !68
  %i61 = getelementptr inbounds %"class.Kripke::Core::DataStore", %"class.Kripke::Core::DataStore"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %i62 = getelementptr inbounds i8, i8* %i61, i64 16
  %i63 = bitcast i8* %i62 to %"struct.std::_Rb_tree_node"**
  %i64 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i65 = getelementptr inbounds i8, i8* %i61, i64 8
  %i66 = bitcast i8* %i65 to %"struct.std::_Rb_tree_node_base"*
  %i71 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i64, i64 0, i32 1, i32 0, i64 8
  %i72 = bitcast i8* %i71 to i64*
  %i73 = load i64, i64* %i72, align 8, !tbaa !69
  %i74 = icmp ult i64 %i73, 6
  %i75 = select i1 %i74, i64 %i73, i64 6
  %i76 = icmp eq i64 %i75, 0
  br i1 %i76, label %bb83, label %bb77

bb77:                                             ; preds = %bb
  %i78 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i64, i64 0, i32 1
  %i79 = bitcast %"struct.__gnu_cxx::__aligned_membuf"* %i78 to i8**
  %i80 = load i8*, i8** %i79, align 8, !tbaa !58
  %i81 = call i32 @memcmp(i8* %i80, i8* nonnull %i57, i64 %i75) #16
  %i82 = icmp eq i32 %i81, 0
  br i1 %i82, label %bb83, label %bb90

bb83:                                             ; preds = %bb77, %bb
  %i84 = add i64 %i73, -6
  %i85 = icmp sgt i64 %i84, 2147483647
  br i1 %i85, label %bb93, label %bb86

bb86:                                             ; preds = %bb83
  %i87 = icmp sgt i64 %i84, -2147483648
  %i88 = select i1 %i87, i64 %i84, i64 -2147483648
  %i89 = trunc i64 %i88 to i32
  br label %bb90

bb90:                                             ; preds = %bb86, %bb77
  %i91 = phi i32 [ %i81, %bb77 ], [ %i89, %bb86 ]
  %i92 = icmp slt i32 %i91, 0
  br i1 %i92, label %bb106, label %bb93

bb93:                                             ; preds = %bb90, %bb83
  %i94 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i64, i64 0, i32 0
  br label %bb106

bb106:                                            ; preds = %bb93, %bb90
  %i99 = phi %"struct.std::_Rb_tree_node_base"* [ %i94, %bb93 ], [ %i66, %bb90 ]
  %i189 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i99, i64 2
  %i18 = bitcast %"struct.std::_Rb_tree_node_base"* %i189 to i8**
  %i1914 = load i8*, i8** %i18, align 8, !tbaa !194
  %i195 = call i8* @__dynamic_cast(i8* nonnull %i1914, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke6TimingE to i8*), i64 0) #16
  %i196 = bitcast i8* %i195 to %"class.Kripke::Timing"*
  %i204 = bitcast %"class.std::__cxx11::basic_string"* %i36 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i204) #16
  %i205 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i36, i64 0, i32 2
  %i206 = bitcast %"class.std::__cxx11::basic_string"* %i36 to %union.anon**
  store %union.anon* %i205, %union.anon** %i206, align 8, !tbaa !67
  %i207 = bitcast %union.anon* %i205 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(10) %i207, i8* nonnull align 1 dereferenceable(10) getelementptr inbounds ([11 x i8], [11 x i8]* @.str.256, i64 0, i64 0), i64 10, i1 false) #16
  %i209 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i36, i64 0, i32 1
  store i64 10, i64* %i209, align 8, !tbaa !69
  %i210 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i36, i64 0, i32 2, i32 1, i64 2
  store i8 0, i8* %i210, align 2, !tbaa !68
  %i19 = bitcast %"class.Kripke::BlockTimer"* %i34 to i8**
  store i8* %i195, i8** %i19, align 8, !tbaa !78
  %i212 = getelementptr inbounds %"class.Kripke::BlockTimer", %"class.Kripke::BlockTimer"* %i34, i64 0, i32 1
  %i213 = getelementptr inbounds %"class.Kripke::BlockTimer", %"class.Kripke::BlockTimer"* %i34, i64 0, i32 1, i32 2
  %i214 = bitcast %"class.std::__cxx11::basic_string"* %i212 to %union.anon**
  store %union.anon* %i213, %union.anon** %i214, align 8, !tbaa !67
  %i215 = bitcast %union.anon* %i213 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(10) %i215, i8* nonnull align 8 dereferenceable(10) %i207, i64 10, i1 false) #16
  %i217 = getelementptr inbounds %"class.Kripke::BlockTimer", %"class.Kripke::BlockTimer"* %i34, i64 0, i32 1, i32 1
  store i64 10, i64* %i217, align 8, !tbaa !69
  %i218 = getelementptr inbounds %"class.Kripke::BlockTimer", %"class.Kripke::BlockTimer"* %i34, i64 0, i32 1, i32 2, i32 1, i64 2
  store i8 0, i8* %i218, align 2, !tbaa !68
  call fastcc void @_ZN6Kripke6Timing5startERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.Kripke::Timing"* nonnull dereferenceable(64) %i196, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %i212)
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i204) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i54) #16
  %i233 = bitcast %"class.std::__cxx11::basic_string"* %i37 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i233) #16
  %i234 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i37, i64 0, i32 2
  %i235 = bitcast %"class.std::__cxx11::basic_string"* %i37 to %union.anon**
  store %union.anon* %i234, %union.anon** %i235, align 8, !tbaa !67
  %i236 = bitcast %union.anon* %i234 to i8*
  %i237 = bitcast %union.anon* %i234 to i16*
  store i16 27745, i16* %i237, align 8
  %i239 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i37, i64 0, i32 1
  store i64 2, i64* %i239, align 8, !tbaa !69
  %i240 = getelementptr inbounds i8, i8* %i236, i64 2
  store i8 0, i8* %i240, align 2, !tbaa !68
  %i241 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i246 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i241, i64 0, i32 1, i32 0, i64 8
  %i247 = bitcast i8* %i246 to i64*
  %i248 = load i64, i64* %i247, align 8, !tbaa !69
  %i249 = icmp ult i64 %i248, 2
  %i250 = select i1 %i249, i64 %i248, i64 2
  %i251 = icmp eq i64 %i250, 0
  br i1 %i251, label %bb258, label %bb252

bb252:                                            ; preds = %bb106
  %i253 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i241, i64 0, i32 1
  %i254 = bitcast %"struct.__gnu_cxx::__aligned_membuf"* %i253 to i8**
  %i255 = load i8*, i8** %i254, align 8, !tbaa !58
  %i256 = call i32 @memcmp(i8* %i255, i8* nonnull %i236, i64 %i250) #16
  %i257 = icmp eq i32 %i256, 0
  br i1 %i257, label %bb258, label %bb265

bb258:                                            ; preds = %bb252, %bb106
  %i259 = add i64 %i248, -2
  %i260 = icmp sgt i64 %i259, 2147483647
  br i1 %i260, label %bb268, label %bb261

bb261:                                            ; preds = %bb258
  %i262 = icmp sgt i64 %i259, -2147483648
  %i263 = select i1 %i262, i64 %i259, i64 -2147483648
  %i264 = trunc i64 %i263 to i32
  br label %bb265

bb265:                                            ; preds = %bb261, %bb252
  %i266 = phi i32 [ %i256, %bb252 ], [ %i264, %bb261 ]
  %i267 = icmp slt i32 %i266, 0
  br i1 %i267, label %bb281, label %bb268

bb268:                                            ; preds = %bb265, %bb258
  %i269 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i241, i64 0, i32 0
  br label %bb281

bb281:                                            ; preds = %bb268, %bb265
  %i274 = phi %"struct.std::_Rb_tree_node_base"* [ %i269, %bb268 ], [ %i66, %bb265 ]
  %i364 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i274, i64 2
  %i20 = bitcast %"struct.std::_Rb_tree_node_base"* %i364 to i8**
  %i3665 = load i8*, i8** %i20, align 8, !tbaa !194
  %i370 = call i8* @__dynamic_cast(i8* nonnull %i3665, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke10ArchLayoutE to i8*), i64 0) #16
  %i373 = getelementptr inbounds i8, i8* %i370, i64 16
  %i374 = bitcast i8* %i373 to i64*
  %i375 = load i64, i64* %i374, align 8, !tbaa.struct !196
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i233) #16
  %i387 = bitcast %"class.std::__cxx11::basic_string"* %i38 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i387) #16
  %i388 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i38, i64 0, i32 2
  %i389 = bitcast %"class.std::__cxx11::basic_string"* %i38 to %union.anon**
  store %union.anon* %i388, %union.anon** %i389, align 8, !tbaa !67
  %i390 = bitcast %union.anon* %i388 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(6) %i390, i8* nonnull align 1 dereferenceable(6) getelementptr inbounds ([7 x i8], [7 x i8]* @.str.5.264, i64 0, i64 0), i64 6, i1 false) #16
  %i392 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i38, i64 0, i32 1
  store i64 6, i64* %i392, align 8, !tbaa !69
  %i393 = getelementptr inbounds i8, i8* %i390, i64 6
  store i8 0, i8* %i393, align 2, !tbaa !68
  %i394 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i422 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i394, i64 0, i32 0
  %i517 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i422, i64 2
  %i21 = bitcast %"struct.std::_Rb_tree_node_base"* %i517 to i8**
  %i5196 = load i8*, i8** %i21, align 8, !tbaa !194
  %i523 = call i8* @__dynamic_cast(i8* nonnull %i5196, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core14PartitionSpaceE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i387) #16
  %i536 = bitcast %"class.std::__cxx11::basic_string"* %i39 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i536) #16
  %i537 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i39, i64 0, i32 2
  %i538 = bitcast %"class.std::__cxx11::basic_string"* %i39 to %union.anon**
  store %union.anon* %i537, %union.anon** %i538, align 8, !tbaa !67
  %i539 = bitcast %union.anon* %i537 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(9) %i539, i8* nonnull align 1 dereferenceable(9) getelementptr inbounds ([10 x i8], [10 x i8]* @.str.6.265, i64 0, i64 0), i64 9, i1 false) #16
  %i541 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i39, i64 0, i32 1
  store i64 9, i64* %i541, align 8, !tbaa !69
  %i542 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i39, i64 0, i32 2, i32 1, i64 1
  store i8 0, i8* %i542, align 1, !tbaa !68
  %i543 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i571 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i543, i64 0, i32 0
  %i666 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i571, i64 2
  %i22 = bitcast %"struct.std::_Rb_tree_node_base"* %i666 to i8**
  %i6687 = load i8*, i8** %i22, align 8, !tbaa !194
  %i672 = call i8* @__dynamic_cast(i8* nonnull %i6687, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core3SetE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i536) #16
  %i685 = bitcast %"class.std::__cxx11::basic_string"* %i40 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i685) #16
  %i686 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i40, i64 0, i32 2
  %i687 = bitcast %"class.std::__cxx11::basic_string"* %i40 to %union.anon**
  store %union.anon* %i686, %union.anon** %i687, align 8, !tbaa !67
  %i688 = bitcast %union.anon* %i686 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(10) %i688, i8* nonnull align 1 dereferenceable(10) getelementptr inbounds ([11 x i8], [11 x i8]* @.str.7.266, i64 0, i64 0), i64 10, i1 false) #16
  %i690 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i40, i64 0, i32 1
  store i64 10, i64* %i690, align 8, !tbaa !69
  %i691 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i40, i64 0, i32 2, i32 1, i64 2
  store i8 0, i8* %i691, align 2, !tbaa !68
  %i692 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i720 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i692, i64 0, i32 0
  %i815 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i720, i64 2
  %i23 = bitcast %"struct.std::_Rb_tree_node_base"* %i815 to i8**
  %i8178 = load i8*, i8** %i23, align 8, !tbaa !194
  %i821 = call i8* @__dynamic_cast(i8* nonnull %i8178, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core3SetE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i685) #16
  %i834 = bitcast %"class.std::__cxx11::basic_string"* %i41 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i834) #16
  %i835 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i41, i64 0, i32 2
  %i836 = bitcast %"class.std::__cxx11::basic_string"* %i41 to %union.anon**
  store %union.anon* %i835, %union.anon** %i836, align 8, !tbaa !67
  %i839 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i41, i64 0, i32 1
  %i840 = bitcast i64* %i839 to <2 x i64>*
  store <2 x i64> <i64 8, i64 7308901678402790739>, <2 x i64>* %i840, align 8
  %i841 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i41, i64 0, i32 2, i32 1, i64 0
  store i8 0, i8* %i841, align 8, !tbaa !68
  %i842 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i870 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i842, i64 0, i32 0
  %i965 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i870, i64 2
  %i24 = bitcast %"struct.std::_Rb_tree_node_base"* %i965 to i8**
  %i9679 = load i8*, i8** %i24, align 8, !tbaa !194
  %i971 = call i8* @__dynamic_cast(i8* nonnull %i9679, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core3SetE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i834) #16
  %i984 = bitcast %"class.std::__cxx11::basic_string"* %i42 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i984) #16
  %i985 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i42, i64 0, i32 2
  %i986 = bitcast %"class.std::__cxx11::basic_string"* %i42 to %union.anon**
  store %union.anon* %i985, %union.anon** %i986, align 8, !tbaa !67
  %i987 = bitcast %union.anon* %i985 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(3) %i987, i8* nonnull align 1 dereferenceable(3) getelementptr inbounds ([4 x i8], [4 x i8]* @.str.9.267, i64 0, i64 0), i64 3, i1 false) #16
  %i989 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i42, i64 0, i32 1
  store i64 3, i64* %i989, align 8, !tbaa !69
  %i990 = getelementptr inbounds i8, i8* %i987, i64 3
  store i8 0, i8* %i990, align 1, !tbaa !68
  %i991 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i1019 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i991, i64 0, i32 0
  %i1114 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i1019, i64 2
  %i25 = bitcast %"struct.std::_Rb_tree_node_base"* %i1114 to i8**
  %i111610 = load i8*, i8** %i25, align 8, !tbaa !194
  %i1120 = call i8* @__dynamic_cast(i8* nonnull %i111610, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i984) #16
  %i1133 = bitcast %"class.std::__cxx11::basic_string"* %i43 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i1133) #16
  %i1134 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i43, i64 0, i32 2
  %i1135 = bitcast %"class.std::__cxx11::basic_string"* %i43 to %union.anon**
  store %union.anon* %i1134, %union.anon** %i1135, align 8, !tbaa !67
  %i1136 = bitcast %union.anon* %i1134 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(7) %i1136, i8* nonnull align 1 dereferenceable(7) getelementptr inbounds ([8 x i8], [8 x i8]* @.str.10.268, i64 0, i64 0), i64 7, i1 false) #16
  %i1138 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i43, i64 0, i32 1
  store i64 7, i64* %i1138, align 8, !tbaa !69
  %i1139 = getelementptr inbounds i8, i8* %i1136, i64 7
  store i8 0, i8* %i1139, align 1, !tbaa !68
  %i1140 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  br label %bb1142

bb1142:                                           ; preds = %bb1172, %bb281
  %i1143 = phi %"struct.std::_Rb_tree_node"* [ %i1176, %bb1172 ], [ %i1140, %bb281 ]
  %i1144 = phi %"struct.std::_Rb_tree_node_base"* [ %i1173, %bb1172 ], [ %i66, %bb281 ]
  %i1145 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1143, i64 0, i32 1, i32 0, i64 8
  %i1146 = bitcast i8* %i1145 to i64*
  %i1147 = load i64, i64* %i1146, align 8, !tbaa !69
  %i1148 = icmp ult i64 %i1147, 7
  %i1149 = select i1 %i1148, i64 %i1147, i64 7
  %i1150 = icmp eq i64 %i1149, 0
  br i1 %i1150, label %bb1157, label %bb1151

bb1151:                                           ; preds = %bb1142
  %i1152 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1143, i64 0, i32 1
  %i1153 = bitcast %"struct.__gnu_cxx::__aligned_membuf"* %i1152 to i8**
  %i1154 = load i8*, i8** %i1153, align 8, !tbaa !58
  %i1155 = call i32 @memcmp(i8* %i1154, i8* nonnull %i1136, i64 %i1149) #16
  %i1156 = icmp eq i32 %i1155, 0
  br i1 %i1156, label %bb1157, label %bb1164

bb1157:                                           ; preds = %bb1151, %bb1142
  %i1158 = add i64 %i1147, -7
  %i1159 = icmp sgt i64 %i1158, 2147483647
  br i1 %i1159, label %bb1167, label %bb1160

bb1160:                                           ; preds = %bb1157
  %i1161 = icmp sgt i64 %i1158, -2147483648
  %i1162 = select i1 %i1161, i64 %i1158, i64 -2147483648
  %i1163 = trunc i64 %i1162 to i32
  br label %bb1164

bb1164:                                           ; preds = %bb1160, %bb1151
  %i1165 = phi i32 [ %i1155, %bb1151 ], [ %i1163, %bb1160 ]
  %i1166 = icmp slt i32 %i1165, 0
  br i1 %i1166, label %bb1170, label %bb1167

bb1167:                                           ; preds = %bb1164, %bb1157
  %i1168 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1143, i64 0, i32 0
  %i1169 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1143, i64 0, i32 0, i32 2
  br label %bb1172

bb1170:                                           ; preds = %bb1164
  %i1171 = getelementptr inbounds %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1143, i64 0, i32 0, i32 3
  br label %bb1172

bb1172:                                           ; preds = %bb1170, %bb1167
  %i1173 = phi %"struct.std::_Rb_tree_node_base"* [ %i1144, %bb1170 ], [ %i1168, %bb1167 ]
  %i1174 = phi %"struct.std::_Rb_tree_node_base"** [ %i1171, %bb1170 ], [ %i1169, %bb1167 ]
  %i1175 = bitcast %"struct.std::_Rb_tree_node_base"** %i1174 to %"struct.std::_Rb_tree_node"**
  %i1176 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i1175, align 8, !tbaa !78
  %i1177 = icmp eq %"struct.std::_Rb_tree_node"* %i1176, null
  br i1 %i1177, label %bb1180, label %bb1142, !llvm.loop !201

bb1180:                                           ; preds = %bb1172
  %i1263 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i1173, i64 2
  %i26 = bitcast %"struct.std::_Rb_tree_node_base"* %i1263 to i8**
  %i126511 = load i8*, i8** %i26, align 8, !tbaa !194
  %i1269 = call i8* @__dynamic_cast(i8* nonnull %i126511, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i1133) #16
  %i1282 = bitcast %"class.std::__cxx11::basic_string"* %i44 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i1282) #16
  %i1283 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i44, i64 0, i32 2
  %i1284 = bitcast %"class.std::__cxx11::basic_string"* %i44 to %union.anon**
  store %union.anon* %i1283, %union.anon** %i1284, align 8, !tbaa !67
  %i1285 = bitcast %union.anon* %i1283 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(9) %i1285, i8* nonnull align 1 dereferenceable(9) getelementptr inbounds ([10 x i8], [10 x i8]* @.str.11.269, i64 0, i64 0), i64 9, i1 false) #16
  %i1287 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i44, i64 0, i32 1
  store i64 9, i64* %i1287, align 8, !tbaa !69
  %i1288 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i44, i64 0, i32 2, i32 1, i64 1
  store i8 0, i8* %i1288, align 1, !tbaa !68
  %i1289 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i1317 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1289, i64 0, i32 0
  %i1412 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i1317, i64 2
  %i27 = bitcast %"struct.std::_Rb_tree_node_base"* %i1412 to i8**
  %i141412 = load i8*, i8** %i27, align 8, !tbaa !194
  %i1418 = call i8* @__dynamic_cast(i8* nonnull %i141412, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i1282) #16
  %i1431 = bitcast %"class.std::__cxx11::basic_string"* %i45 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i1431) #16
  %i1432 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i45, i64 0, i32 2
  %i1433 = bitcast %"class.std::__cxx11::basic_string"* %i45 to %union.anon**
  store %union.anon* %i1432, %union.anon** %i1433, align 8, !tbaa !67
  %i1434 = bitcast %union.anon* %i1432 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(15) %i1434, i8* nonnull align 1 dereferenceable(15) getelementptr inbounds ([16 x i8], [16 x i8]* @.str.12.270, i64 0, i64 0), i64 15, i1 false) #16
  %i1436 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i45, i64 0, i32 1
  store i64 15, i64* %i1436, align 8, !tbaa !69
  %i1437 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i45, i64 0, i32 2, i32 1, i64 7
  store i8 0, i8* %i1437, align 1, !tbaa !68
  %i1438 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i1466 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1438, i64 0, i32 0
  %i1561 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i1466, i64 2
  %i28 = bitcast %"struct.std::_Rb_tree_node_base"* %i1561 to i8**
  %i156313 = load i8*, i8** %i28, align 8, !tbaa !194
  %i1567 = call i8* @__dynamic_cast(i8* nonnull %i156313, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i1431) #16
  %i1580 = bitcast %"class.std::__cxx11::basic_string"* %i46 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i1580) #16
  %i1581 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i46, i64 0, i32 2
  %i1582 = bitcast %"class.std::__cxx11::basic_string"* %i46 to %union.anon**
  store %union.anon* %i1581, %union.anon** %i1582, align 8, !tbaa !67
  %i1584 = bitcast i64* %i13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i1584) #16
  store i64 19, i64* %i13, align 8, !tbaa !191
  %i1585 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %i46, i64* nonnull align 8 dereferenceable(8) %i13, i64 0)
  %i1587 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i46, i64 0, i32 0, i32 0
  store i8* %i1585, i8** %i1587, align 8, !tbaa !58
  %i1588 = load i64, i64* %i13, align 8, !tbaa !191
  %i1589 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i46, i64 0, i32 2, i32 0
  store i64 %i1588, i64* %i1589, align 8, !tbaa !68
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 dereferenceable(19) %i1585, i8* nonnull align 1 dereferenceable(19) getelementptr inbounds ([20 x i8], [20 x i8]* @.str.13.271, i64 0, i64 0), i64 19, i1 false) #16
  %i1590 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i46, i64 0, i32 1
  store i64 %i1588, i64* %i1590, align 8, !tbaa !69
  %i1591 = load i8*, i8** %i1587, align 8, !tbaa !58
  %i1592 = getelementptr inbounds i8, i8* %i1591, i64 %i1588
  store i8 0, i8* %i1592, align 1, !tbaa !68
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i1584) #16
  %i1593 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i1624 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1593, i64 0, i32 0
  %i1719 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i1624, i64 2
  %i29 = bitcast %"struct.std::_Rb_tree_node_base"* %i1719 to i8**
  %i172114 = load i8*, i8** %i29, align 8, !tbaa !194
  %i1725 = call i8* @__dynamic_cast(i8* nonnull %i172114, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIiJNS_4ZoneEEEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i1580) #16
  %i1740 = bitcast %"class.std::__cxx11::basic_string"* %i47 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i1740) #16
  %i1741 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i47, i64 0, i32 2
  %i1742 = bitcast %"class.std::__cxx11::basic_string"* %i47 to %union.anon**
  store %union.anon* %i1741, %union.anon** %i1742, align 8, !tbaa !67
  %i1744 = bitcast i64* %i10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i1744) #16
  store i64 19, i64* %i10, align 8, !tbaa !191
  %i1745 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %i47, i64* nonnull align 8 dereferenceable(8) %i10, i64 0)
  %i1747 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i47, i64 0, i32 0, i32 0
  store i8* %i1745, i8** %i1747, align 8, !tbaa !58
  %i1748 = load i64, i64* %i10, align 8, !tbaa !191
  %i1749 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i47, i64 0, i32 2, i32 0
  store i64 %i1748, i64* %i1749, align 8, !tbaa !68
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 dereferenceable(19) %i1745, i8* nonnull align 1 dereferenceable(19) getelementptr inbounds ([20 x i8], [20 x i8]* @.str.14.272, i64 0, i64 0), i64 19, i1 false) #16
  %i1750 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i47, i64 0, i32 1
  store i64 %i1748, i64* %i1750, align 8, !tbaa !69
  %i1751 = load i8*, i8** %i1747, align 8, !tbaa !58
  %i1752 = getelementptr inbounds i8, i8* %i1751, i64 %i1748
  store i8 0, i8* %i1752, align 1, !tbaa !68
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i1744) #16
  %i1753 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i1784 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1753, i64 0, i32 0
  %i1879 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i1784, i64 2
  %i30 = bitcast %"struct.std::_Rb_tree_node_base"* %i1879 to i8**
  %i188115 = load i8*, i8** %i30, align 8, !tbaa !194
  %i1885 = call i8* @__dynamic_cast(i8* nonnull %i188115, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i1740) #16
  %i1900 = bitcast %"class.std::__cxx11::basic_string"* %i48 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i1900) #16
  %i1901 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i48, i64 0, i32 2
  %i1902 = bitcast %"class.std::__cxx11::basic_string"* %i48 to %union.anon**
  store %union.anon* %i1901, %union.anon** %i1902, align 8, !tbaa !67
  %i1904 = bitcast i64* %i7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i1904) #16
  store i64 19, i64* %i7, align 8, !tbaa !191
  %i1905 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %i48, i64* nonnull align 8 dereferenceable(8) %i7, i64 0)
  %i1907 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i48, i64 0, i32 0, i32 0
  store i8* %i1905, i8** %i1907, align 8, !tbaa !58
  %i1908 = load i64, i64* %i7, align 8, !tbaa !191
  %i1909 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i48, i64 0, i32 2, i32 0
  store i64 %i1908, i64* %i1909, align 8, !tbaa !68
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 dereferenceable(19) %i1905, i8* nonnull align 1 dereferenceable(19) getelementptr inbounds ([20 x i8], [20 x i8]* @.str.15.273, i64 0, i64 0), i64 19, i1 false) #16
  %i1910 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i48, i64 0, i32 1
  store i64 %i1908, i64* %i1910, align 8, !tbaa !69
  %i1911 = load i8*, i8** %i1907, align 8, !tbaa !58
  %i1912 = getelementptr inbounds i8, i8* %i1911, i64 %i1908
  store i8 0, i8* %i1912, align 1, !tbaa !68
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i1904) #16
  %i1913 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i1944 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i1913, i64 0, i32 0
  %i2039 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i1944, i64 2
  %i31 = bitcast %"struct.std::_Rb_tree_node_base"* %i2039 to i8**
  %i204116 = load i8*, i8** %i31, align 8, !tbaa !194
  %i2045 = call i8* @__dynamic_cast(i8* nonnull %i204116, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldIdJNS_7MixElemEEEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i1900) #16
  %i2060 = bitcast %"class.std::__cxx11::basic_string"* %i49 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %i2060) #16
  %i2061 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i49, i64 0, i32 2
  %i2062 = bitcast %"class.std::__cxx11::basic_string"* %i49 to %union.anon**
  store %union.anon* %i2061, %union.anon** %i2062, align 8, !tbaa !67
  %i2064 = bitcast i64* %i4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i2064) #16
  store i64 18, i64* %i4, align 8, !tbaa !191
  %i2065 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %i49, i64* nonnull align 8 dereferenceable(8) %i4, i64 0)
  %i2067 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i49, i64 0, i32 0, i32 0
  store i8* %i2065, i8** %i2067, align 8, !tbaa !58
  %i2068 = load i64, i64* %i4, align 8, !tbaa !191
  %i2069 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i49, i64 0, i32 2, i32 0
  store i64 %i2068, i64* %i2069, align 8, !tbaa !68
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 dereferenceable(18) %i2065, i8* nonnull align 1 dereferenceable(18) getelementptr inbounds ([19 x i8], [19 x i8]* @.str.16.274, i64 0, i64 0), i64 18, i1 false) #16
  %i2070 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i49, i64 0, i32 1
  store i64 %i2068, i64* %i2070, align 8, !tbaa !69
  %i2071 = load i8*, i8** %i2067, align 8, !tbaa !58
  %i2072 = getelementptr inbounds i8, i8* %i2071, i64 %i2068
  store i8 0, i8* %i2072, align 1, !tbaa !68
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i2064) #16
  %i2073 = load %"struct.std::_Rb_tree_node"*, %"struct.std::_Rb_tree_node"** %i63, align 8, !tbaa !192
  %i2104 = getelementptr %"struct.std::_Rb_tree_node", %"struct.std::_Rb_tree_node"* %i2073, i64 0, i32 0
  %i2199 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i2104, i64 2
  %i32 = bitcast %"struct.std::_Rb_tree_node_base"* %i2199 to i8**
  %i220117 = load i8*, i8** %i32, align 8, !tbaa !194
  %i2205 = call i8* @__dynamic_cast(i8* nonnull %i220117, i8* bitcast ({ i8*, i8* }* @_ZTIN6Kripke4Core7BaseVarE to i8*), i8* bitcast ({ i8*, i8*, i8* }* @_ZTIN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE to i8*), i64 0) #16
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %i2060) #16
  %i2220 = getelementptr inbounds i8, i8* %i1120, i64 64
  %i33 = bitcast i8* %i2220 to %"class.Kripke::SdomId"**
  %i2221 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i33, align 8, !tbaa !78
  %i2226 = bitcast %"class.Kripke::SdomId"* %i50 to i8*
  %i2227 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i50, i64 0, i32 0, i32 0
  %i2230 = bitcast %"class.Kripke::SdomId"* %i51 to i8*
  %i2231 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i51, i64 0, i32 0, i32 0
  %i2232 = getelementptr inbounds %"class.std::ios_base::Init", %"class.std::ios_base::Init"* %i52, i64 0, i32 0
  %i2233 = bitcast i64* %i to i8*
  %i2235 = bitcast %class.anon.509* %i1 to i8*
  %i2236 = bitcast %class.anon.509* %i1 to i64**
  %i2237 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 1
  %i2238 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 2
  %i2239 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 3
  %i2240 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 4
  %i2241 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 5
  %i2242 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 6
  %i2243 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 7
  %i2244 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 8
  %i2245 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 9
  %i2246 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 10
  %i2247 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 11
  %i2248 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 12
  %i2249 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 13
  %i2250 = getelementptr inbounds %class.anon.509, %class.anon.509* %i1, i64 0, i32 14
  %i2229 = getelementptr inbounds i8, i8* %i1269, i64 72
  %i77 = bitcast i8* %i2229 to %"class.Kripke::SdomId"**
  %i2228 = getelementptr inbounds i8, i8* %i1269, i64 64
  %i68 = bitcast i8* %i2228 to %"class.Kripke::SdomId"**
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i2226) #16
  %i2392 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i2221, i64 0, i32 0, i32 0
  %i2393 = load i64, i64* %i2392, align 8
  store i64 %i2393, i64* %i2227, align 8
  %i2394 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i68, align 8, !tbaa !78
  %i2395 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i77, align 8, !tbaa !78
  br label %bb2398

bb2398:                                           ; preds = %bb2398, %bb1180
  %i2400 = phi %"class.Kripke::SdomId"* [ %i2394, %bb1180 ], [ %i2411, %bb2398 ]
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i2230) #16
  %i2401 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i2400, i64 0, i32 0, i32 0
  %i2402 = load i64, i64* %i2401, align 8
  store i64 %i2402, i64* %i2231, align 8
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %i2232) #16
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i2233)
  store i64 %i375, i64* %i, align 8
  call void @llvm.lifetime.start.p0i8(i64 120, i8* nonnull %i2235) #16
  store i64* %i, i64** %i2236, align 8, !tbaa !78
  store %"class.std::ios_base::Init"* %i52, %"class.std::ios_base::Init"** %i2237, align 8, !tbaa !78
  store %"class.Kripke::SdomId"* %i50, %"class.Kripke::SdomId"** %i2238, align 8, !tbaa !78
  store %"class.Kripke::SdomId"* %i51, %"class.Kripke::SdomId"** %i2239, align 8, !tbaa !78
  %i83 = bitcast %"class.Kripke::Core::Set"** %i2240 to i8**
  store i8* %i672, i8** %i83, align 8, !tbaa !78
  %i86 = bitcast %"class.Kripke::Core::Set"** %i2241 to i8**
  store i8* %i971, i8** %i86, align 8, !tbaa !78
  %i90 = bitcast %"class.Kripke::Core::Set"** %i2242 to i8**
  store i8* %i821, i8** %i90, align 8, !tbaa !78
  %i93 = bitcast %"class.Kripke::Core::Field"** %i2243 to i8**
  store i8* %i1120, i8** %i93, align 8, !tbaa !78
  %i96 = bitcast %"class.Kripke::Core::Field"** %i2244 to i8**
  store i8* %i1269, i8** %i96, align 8, !tbaa !78
  %i98 = bitcast %"class.Kripke::Core::Field.33"** %i2245 to i8**
  store i8* %i1418, i8** %i98, align 8, !tbaa !78
  %i104 = bitcast %"class.Kripke::Core::Field.246"** %i2246 to i8**
  store i8* %i1567, i8** %i104, align 8, !tbaa !78
  %i105 = bitcast %"class.Kripke::Core::Field.48.258"** %i2247 to i8**
  store i8* %i1725, i8** %i105, align 8, !tbaa !78
  %i106 = bitcast %"class.Kripke::Core::Field.246"** %i2248 to i8**
  store i8* %i1885, i8** %i106, align 8, !tbaa !78
  %i107 = bitcast %"class.Kripke::Core::Field.35"** %i2249 to i8**
  store i8* %i2045, i8** %i107, align 8, !tbaa !78
  %i108 = bitcast %"class.Kripke::Core::Field.246"** %i2250 to i8**
  store i8* %i2205, i8** %i108, align 8, !tbaa !78
  call fastcc void @_ZZN6Kripke8dispatchI14ScatteringSdomJRNS_6SdomIdES3_RNS_4Core3SetES6_S6_RNS4_5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEESC_RNS7_IdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESF_EEERNS7_INS_7MixElemEJSA_EEERNS7_IiJSA_EEERNS7_ISD_JSI_EEERNS7_IdJSI_EEERNS7_ISE_JS8_EEEEEEvNS_11ArchLayoutVERKT_DpOT0_ENKUlSU_E_clINS_16ArchT_SequentialEEEDaSU_(%class.anon.509* nonnull dereferenceable(120) %i1)
  call void @llvm.lifetime.end.p0i8(i64 120, i8* nonnull %i2235) #16
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i2233)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %i2232) #16
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i2230) #16
  %i2411 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i2400, i64 1
  %i2412 = icmp eq %"class.Kripke::SdomId"* %i2411, %i2395
  br i1 %i2412, label %bb2444, label %bb2398, !llvm.loop !202

bb2444:                                           ; preds = %bb2398
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i2226) #16
  call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %i53) #16
  ret void
}

; Function Attrs: inlinehint uwtable mustprogress
define internal fastcc void @_ZZN6Kripke8dispatchI14ScatteringSdomJRNS_6SdomIdES3_RNS_4Core3SetES6_S6_RNS4_5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEESC_RNS7_IdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESF_EEERNS7_INS_7MixElemEJSA_EEERNS7_IiJSA_EEERNS7_ISD_JSI_EEERNS7_IdJSI_EEERNS7_ISE_JS8_EEEEEEvNS_11ArchLayoutVERKT_DpOT0_ENKUlSU_E_clINS_16ArchT_SequentialEEEDaSU_(%class.anon.509* nonnull dereferenceable(120) %arg) unnamed_addr #15 comdat align 2 {
bb:
  %i = alloca %"class.std::ios_base::Init", align 1
  %i1 = getelementptr inbounds %"class.std::ios_base::Init", %"class.std::ios_base::Init"* %i, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %i1) #16
  %i6 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 1
  %i7 = load %"class.std::ios_base::Init"*, %"class.std::ios_base::Init"** %i6, align 8, !tbaa !203
  %i8 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 2
  %i9 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i8, align 8, !tbaa !205
  %i10 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 3
  %i11 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i10, align 8, !tbaa !206
  %i12 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 4
  %i13 = load %"class.Kripke::Core::Set"*, %"class.Kripke::Core::Set"** %i12, align 8, !tbaa !207
  %i14 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 5
  %i15 = load %"class.Kripke::Core::Set"*, %"class.Kripke::Core::Set"** %i14, align 8, !tbaa !208
  %i16 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 6
  %i17 = load %"class.Kripke::Core::Set"*, %"class.Kripke::Core::Set"** %i16, align 8, !tbaa !209
  %i18 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 7
  %i19 = load %"class.Kripke::Core::Field"*, %"class.Kripke::Core::Field"** %i18, align 8, !tbaa !210
  %i20 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 8
  %i21 = load %"class.Kripke::Core::Field"*, %"class.Kripke::Core::Field"** %i20, align 8, !tbaa !211
  %i22 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 9
  %i23 = load %"class.Kripke::Core::Field.33"*, %"class.Kripke::Core::Field.33"** %i22, align 8, !tbaa !212
  %i24 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 10
  %i25 = load %"class.Kripke::Core::Field.246"*, %"class.Kripke::Core::Field.246"** %i24, align 8, !tbaa !213
  %i26 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 11
  %i27 = load %"class.Kripke::Core::Field.48.258"*, %"class.Kripke::Core::Field.48.258"** %i26, align 8, !tbaa !214
  %i28 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 12
  %i29 = load %"class.Kripke::Core::Field.246"*, %"class.Kripke::Core::Field.246"** %i28, align 8, !tbaa !215
  %i30 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 13
  %i31 = load %"class.Kripke::Core::Field.35"*, %"class.Kripke::Core::Field.35"** %i30, align 8, !tbaa !216
  %i32 = getelementptr inbounds %class.anon.509, %class.anon.509* %arg, i64 0, i32 14
  %i33 = load %"class.Kripke::Core::Field.246"*, %"class.Kripke::Core::Field.246"** %i32, align 8, !tbaa !217
  call fastcc void @_ZNK6Kripke14DispatchHelperINS_16ArchT_SequentialEEclINS_11LayoutT_DZGE14ScatteringSdomJRNS_6SdomIdES7_RNS_4Core3SetESA_SA_RNS8_5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEESG_RNSB_IdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESJ_EEERNSB_INS_7MixElemEJSE_EEERNSB_IiJSE_EEERNSB_ISH_JSM_EEERNSB_IdJSM_EEERNSB_ISI_JSC_EEEEEEvT_RKT0_DpOT1_(%"class.std::ios_base::Init"* nonnull dereferenceable(1) %i, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %i7, %"class.Kripke::SdomId"* nonnull align 8 dereferenceable(8) %i9, %"class.Kripke::SdomId"* nonnull align 8 dereferenceable(8) %i11, %"class.Kripke::Core::Set"* nonnull align 8 dereferenceable(144) %i13, %"class.Kripke::Core::Set"* nonnull align 8 dereferenceable(144) %i15, %"class.Kripke::Core::Set"* nonnull align 8 dereferenceable(144) %i17, %"class.Kripke::Core::Field"* nonnull align 8 dereferenceable(168) %i19, %"class.Kripke::Core::Field"* nonnull align 8 dereferenceable(168) %i21, %"class.Kripke::Core::Field.33"* nonnull align 8 dereferenceable(168) %i23, %"class.Kripke::Core::Field.246"* nonnull align 8 dereferenceable(168) %i25, %"class.Kripke::Core::Field.48.258"* nonnull align 8 dereferenceable(168) %i27, %"class.Kripke::Core::Field.246"* nonnull align 8 dereferenceable(168) %i29, %"class.Kripke::Core::Field.35"* nonnull align 8 dereferenceable(168) %i31, %"class.Kripke::Core::Field.246"* nonnull align 8 dereferenceable(168) %i33)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %i1) #16
  ret void
}

; Function Attrs: uwtable
define internal fastcc void @_ZNK6Kripke14DispatchHelperINS_16ArchT_SequentialEEclINS_11LayoutT_DZGE14ScatteringSdomJRNS_6SdomIdES7_RNS_4Core3SetESA_SA_RNS8_5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEESG_RNSB_IdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESJ_EEERNSB_INS_7MixElemEJSE_EEERNSB_IiJSE_EEERNSB_ISH_JSM_EEERNSB_IdJSM_EEERNSB_ISI_JSC_EEEEEEvT_RKT0_DpOT1_(%"class.std::ios_base::Init"* nonnull dereferenceable(1) %arg, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg1, %"class.Kripke::SdomId"* nonnull align 8 dereferenceable(8) %arg2, %"class.Kripke::SdomId"* nonnull align 8 dereferenceable(8) %arg3, %"class.Kripke::Core::Set"* nonnull align 8 dereferenceable(144) %arg4, %"class.Kripke::Core::Set"* nonnull align 8 dereferenceable(144) %arg5, %"class.Kripke::Core::Set"* nonnull align 8 dereferenceable(144) %arg6, %"class.Kripke::Core::Field"* nonnull align 8 dereferenceable(168) %arg7, %"class.Kripke::Core::Field"* nonnull align 8 dereferenceable(168) %arg8, %"class.Kripke::Core::Field.33"* nonnull align 8 dereferenceable(168) %arg9, %"class.Kripke::Core::Field.246"* nonnull align 8 dereferenceable(168) %arg10, %"class.Kripke::Core::Field.48.258"* nonnull align 8 dereferenceable(168) %arg11, %"class.Kripke::Core::Field.246"* nonnull align 8 dereferenceable(168) %arg12, %"class.Kripke::Core::Field.35"* nonnull align 8 dereferenceable(168) %arg13, %"class.Kripke::Core::Field.246"* nonnull align 8 dereferenceable(168) %arg14) unnamed_addr #9 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i15 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %arg2, i64 0, i32 0, i32 0
  %i16 = load i64, i64* %i15, align 8
  %i17 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %arg3, i64 0, i32 0, i32 0
  %i18 = load i64, i64* %i17, align 8
  %i19 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg4, i64 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i20 = load i64*, i64** %i19, align 8, !tbaa !82
  %i21 = getelementptr inbounds i64, i64* %i20, i64 %i16
  %i22 = load i64, i64* %i21, align 8, !tbaa !191
  %i23 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg4, i64 0, i32 2, i32 0, i32 0, i32 0
  %i24 = load i64*, i64** %i23, align 8, !tbaa !82
  %i25 = getelementptr inbounds i64, i64* %i20, i64 %i18
  %i26 = load i64, i64* %i25, align 8, !tbaa !191
  %i27 = getelementptr inbounds i64, i64* %i24, i64 %i26
  %i28 = load i64, i64* %i27, align 8, !tbaa !191
  %i29 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg14, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i30 = load i64*, i64** %i29, align 8, !tbaa !82, !noalias !218
  %i31 = getelementptr inbounds i64, i64* %i30, i64 %i16
  %i32 = load i64, i64* %i31, align 8, !tbaa !191, !noalias !218
  %i33 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg14, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i34 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i33, align 8, !tbaa !140, !noalias !218
  %i35 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i34, i64 %i32
  %i36 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i35, align 8, !tbaa !78, !noalias !218
  %i37 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg7, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i38 = load i64*, i64** %i37, align 8, !tbaa !82, !noalias !223
  %i39 = getelementptr inbounds i64, i64* %i38, i64 %i16
  %i40 = load i64, i64* %i39, align 8, !tbaa !191, !noalias !223
  %i41 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg7, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i42 = load double**, double*** %i41, align 8, !tbaa !79, !noalias !223
  %i43 = getelementptr inbounds double*, double** %i42, i64 %i40
  %i44 = load double*, double** %i43, align 8, !tbaa !78, !noalias !223
  %i51 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg8, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i52 = load i64*, i64** %i51, align 8, !tbaa !82, !noalias !228
  %i53 = getelementptr inbounds i64, i64* %i52, i64 %i18
  %i54 = load i64, i64* %i53, align 8, !tbaa !191, !noalias !228
  %i55 = getelementptr inbounds %"class.Kripke::Core::Field", %"class.Kripke::Core::Field"* %arg8, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i56 = load double**, double*** %i55, align 8, !tbaa !79, !noalias !228
  %i57 = getelementptr inbounds double*, double** %i56, i64 %i54
  %i58 = load double*, double** %i57, align 8, !tbaa !78, !noalias !228
  %i65 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg9, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i66 = load i64*, i64** %i65, align 8, !tbaa !82, !noalias !233
  %i67 = getelementptr inbounds i64, i64* %i66, i64 %i16
  %i68 = load i64, i64* %i67, align 8, !tbaa !191, !noalias !233
  %i69 = getelementptr inbounds %"class.Kripke::Core::Field.33", %"class.Kripke::Core::Field.33"* %arg9, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i70 = load double**, double*** %i69, align 8, !tbaa !79, !noalias !233
  %i71 = getelementptr inbounds double*, double** %i70, i64 %i68
  %i72 = load double*, double** %i71, align 8, !tbaa !78, !noalias !233
  %i81 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg10, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i82 = load i64*, i64** %i81, align 8, !tbaa !82, !noalias !238
  %i83 = getelementptr inbounds i64, i64* %i82, i64 %i16
  %i84 = load i64, i64* %i83, align 8, !tbaa !191, !noalias !238
  %i85 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg10, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i86 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i85, align 8, !tbaa !150, !noalias !238
  %i87 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i86, i64 %i84
  %i88 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i87, align 8, !tbaa !78, !noalias !238
  %i89 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg11, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i90 = load i64*, i64** %i89, align 8, !tbaa !82, !noalias !243
  %i91 = getelementptr inbounds i64, i64* %i90, i64 %i16
  %i92 = load i64, i64* %i91, align 8, !tbaa !191, !noalias !243
  %i93 = getelementptr inbounds %"class.Kripke::Core::Field.48.258", %"class.Kripke::Core::Field.48.258"* %arg11, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i94 = load i32**, i32*** %i93, align 8, !tbaa !133, !noalias !243
  %i95 = getelementptr inbounds i32*, i32** %i94, i64 %i92
  %i96 = load i32*, i32** %i95, align 8, !tbaa !78, !noalias !243
  %i97 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg12, i64 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0
  %i98 = load i64*, i64** %i97, align 8, !tbaa !82, !noalias !248
  %i99 = getelementptr inbounds i64, i64* %i98, i64 %i16
  %i100 = load i64, i64* %i99, align 8, !tbaa !191, !noalias !248
  %i101 = getelementptr inbounds %"class.Kripke::Core::Field.246", %"class.Kripke::Core::Field.246"* %arg12, i64 0, i32 0, i32 3, i32 0, i32 0, i32 0
  %i102 = load %"class.Kripke::SdomId"**, %"class.Kripke::SdomId"*** %i101, align 8, !tbaa !160, !noalias !248
  %i103 = getelementptr inbounds %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i102, i64 %i100
  %i104 = load %"class.Kripke::SdomId"*, %"class.Kripke::SdomId"** %i103, align 8, !tbaa !78, !noalias !248
  %i113 = getelementptr inbounds %"class.Kripke::Core::Set", %"class.Kripke::Core::Set"* %arg4, i64 0, i32 1, i32 0, i32 0, i32 0
  %i114 = load i64*, i64** %i113, align 8, !tbaa !82
  %i115 = getelementptr inbounds i64, i64* %i114, i64 %i22
  %i116 = load i64, i64* %i115, align 8, !tbaa !191
  %i417 = shl i64 %i28, 32
  %i418 = ashr exact i64 %i417, 32
  br label %bb419

bb419:                                            ; preds = %bb533, %bb
  %i420 = phi i64 [ %i534, %bb533 ], [ 0, %bb ]
  %i429 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i36, i64 %i420, i32 0, i32 0
  br label %bb433

bb433:                                            ; preds = %bb496, %bb419
  %i434 = phi i64 [ %i497, %bb496 ], [ 0, %bb419 ]
  %i442 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i88, i64 %i434, i32 0, i32 0
  %i443 = getelementptr inbounds i32, i32* %i96, i64 %i434
  %i445 = add nuw i64 %i434, %i420
  %i447 = add nuw i64 %i434, %i420
  br label %bb448

bb448:                                            ; preds = %bb493, %bb433
  %i449 = phi i64 [ %i494, %bb493 ], [ 0, %bb433 ]
  %i457 = add nsw i64 %i449, %i418
  %i458 = load i32, i32* %i443, align 4, !tbaa !110
  %i459 = sext i32 %i458 to i64
  %i463 = add i64 %i445, %i449
  %i464 = getelementptr inbounds double, double* %i44, i64 %i463
  %i465 = add i64 %i447, %i449
  %i466 = getelementptr inbounds double, double* %i58, i64 %i465
  br label %bb467

bb467:                                            ; preds = %bb486, %bb448
  %i468 = phi i64 [ %i491, %bb486 ], [ 0, %bb448 ]
  %i469 = load i64, i64* %i442, align 8
  %i470 = add nsw i64 %i469, %i459
  %i471 = load i64, i64* %i429, align 8
  %i473 = add i64 %i457, %i471
  br label %bb474

bb474:                                            ; preds = %bb474, %bb467
  %i475 = phi double [ 0.000000e+00, %bb467 ], [ %i483, %bb474 ]
  %i476 = phi i64 [ %i469, %bb467 ], [ %i484, %bb474 ]
  %i477 = getelementptr inbounds %"class.Kripke::SdomId", %"class.Kripke::SdomId"* %i104, i64 %i476, i32 0, i32 0
  %i478 = load i64, i64* %i477, align 8
  %i480 = add i64 %i473, %i478
  %i481 = getelementptr inbounds double, double* %i72, i64 %i480
  %i482 = load double, double* %i481, align 8, !tbaa !108
  %i483 = fadd fast double %i482, %i475
  %i484 = add nsw i64 %i476, 1
  %i485 = icmp slt i64 %i484, %i470
  br i1 %i485, label %bb474, label %bb486, !llvm.loop !253

bb486:                                            ; preds = %bb474
  %i487 = load double, double* %i464, align 8, !tbaa !108
  %i488 = fmul fast double %i487, %i483
  %i489 = load double, double* %i466, align 8, !tbaa !108
  %i490 = fadd fast double %i489, %i488
  store double %i490, double* %i466, align 8, !tbaa !108
  %i491 = add nuw nsw i64 %i468, 1
  %i492 = icmp eq i64 %i491, %i116
  br i1 %i492, label %bb493, label %bb467, !llvm.loop !254

bb493:                                            ; preds = %bb486
  %i494 = add nuw nsw i64 %i449, 1
  %i495 = icmp eq i64 %i494, 134
  br i1 %i495, label %bb496, label %bb448, !llvm.loop !255

bb496:                                            ; preds = %bb493
  %i497 = add nuw nsw i64 %i434, 1
  %i498 = icmp eq i64 %i497, 142
  br i1 %i498, label %bb533, label %bb433, !llvm.loop !256

bb533:                                            ; preds = %bb496
  %i534 = add nuw nsw i64 %i420, 1
  %i535 = icmp eq i64 %i534, 10
  br i1 %i535, label %bb536, label %bb419, !llvm.loop !257

bb536:                                            ; preds = %bb533
  ret void
}

; Function Attrs: uwtable
define internal fastcc void @_ZN6Kripke6Timing5startERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE(%"class.Kripke::Timing"* nonnull dereferenceable(64) %arg, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg1) unnamed_addr #9 align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Timing", %"class.Kripke::Timing"* %arg, i64 0, i32 1
  %i2 = tail call fastcc nonnull align 8 dereferenceable(48) %"class.Kripke::Timer"* @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEN6Kripke5TimerESt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_(%"class.std::map"* nonnull dereferenceable(48) %i, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg1)
  %i3 = tail call i64 @_ZNSt6chrono3_V212steady_clock3nowEv() #16
  %i4 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i2, i64 0, i32 3, i32 0, i32 1, i32 0, i32 0
  store i64 %i3, i64* %i4, align 8, !tbaa.struct !258
  %i5 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i2, i64 0, i32 3, i32 0, i32 0, i32 0, i32 0
  %i6 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i2, i64 0, i32 3, i32 0, i32 2
  store double 0.000000e+00, double* %i6, align 8, !tbaa !259
  %i7 = tail call i64 @_ZNSt6chrono3_V212steady_clock3nowEv() #16
  store i64 %i7, i64* %i5, align 8, !tbaa.struct !258
  %i8 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i2, i64 0, i32 0
  store i8 1, i8* %i8, align 8, !tbaa !263
  %i9 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i2, i64 0, i32 2
  %i10 = load i64, i64* %i9, align 8, !tbaa !267
  %i11 = add i64 %i10, 1
  store i64 %i11, i64* %i9, align 8, !tbaa !267
  ret void
}

; Function Attrs: uwtable
define internal fastcc nonnull align 8 dereferenceable(48) %"class.Kripke::Timer"* @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEN6Kripke5TimerESt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_(%"class.std::map"* nonnull dereferenceable(48) %arg, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg1) unnamed_addr #9 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = alloca %"class.std::tuple", align 8
  %i2 = alloca %"class.std::ios_base::Init", align 1
  %i3 = getelementptr inbounds %"class.std::map", %"class.std::map"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %i4 = getelementptr inbounds i8, i8* %i3, i64 16
  %i5 = bitcast i8* %i4 to %"struct.std::_Rb_tree_node.542"**
  %i6 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i5, align 8, !tbaa !192
  %i7 = getelementptr inbounds i8, i8* %i3, i64 8
  %i8 = bitcast i8* %i7 to %"struct.std::_Rb_tree_node_base"*
  %i9 = icmp eq %"struct.std::_Rb_tree_node.542"* %i6, null
  br i1 %i9, label %bb76, label %bb10

bb10:                                             ; preds = %bb
  %i11 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 1
  %i12 = load i64, i64* %i11, align 8, !tbaa !69
  %i13 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 0, i32 0
  %i14 = load i8*, i8** %i13, align 8
  br label %bb15

bb15:                                             ; preds = %bb45, %bb10
  %i16 = phi %"struct.std::_Rb_tree_node.542"* [ %i6, %bb10 ], [ %i49, %bb45 ]
  %i17 = phi %"struct.std::_Rb_tree_node_base"* [ %i8, %bb10 ], [ %i46, %bb45 ]
  %i18 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i16, i64 0, i32 1, i32 0, i64 8
  %i19 = bitcast i8* %i18 to i64*
  %i20 = load i64, i64* %i19, align 8, !tbaa !69
  %i21 = icmp ugt i64 %i20, %i12
  %i22 = select i1 %i21, i64 %i12, i64 %i20
  %i23 = icmp eq i64 %i22, 0
  br i1 %i23, label %bb30, label %bb24

bb24:                                             ; preds = %bb15
  %i25 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i16, i64 0, i32 1
  %i26 = bitcast %"struct.__gnu_cxx::__aligned_membuf.541"* %i25 to i8**
  %i27 = load i8*, i8** %i26, align 8, !tbaa !58
  %i28 = tail call i32 @memcmp(i8* %i27, i8* %i14, i64 %i22) #16
  %i29 = icmp eq i32 %i28, 0
  br i1 %i29, label %bb30, label %bb37

bb30:                                             ; preds = %bb24, %bb15
  %i31 = sub i64 %i20, %i12
  %i32 = icmp sgt i64 %i31, 2147483647
  br i1 %i32, label %bb40, label %bb33

bb33:                                             ; preds = %bb30
  %i34 = icmp sgt i64 %i31, -2147483648
  %i35 = select i1 %i34, i64 %i31, i64 -2147483648
  %i36 = trunc i64 %i35 to i32
  br label %bb37

bb37:                                             ; preds = %bb33, %bb24
  %i38 = phi i32 [ %i28, %bb24 ], [ %i36, %bb33 ]
  %i39 = icmp slt i32 %i38, 0
  br i1 %i39, label %bb43, label %bb40

bb40:                                             ; preds = %bb37, %bb30
  %i41 = getelementptr %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i16, i64 0, i32 0
  %i42 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i16, i64 0, i32 0, i32 2
  br label %bb45

bb43:                                             ; preds = %bb37
  %i44 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i16, i64 0, i32 0, i32 3
  br label %bb45

bb45:                                             ; preds = %bb43, %bb40
  %i46 = phi %"struct.std::_Rb_tree_node_base"* [ %i17, %bb43 ], [ %i41, %bb40 ]
  %i47 = phi %"struct.std::_Rb_tree_node_base"** [ %i44, %bb43 ], [ %i42, %bb40 ]
  %i48 = bitcast %"struct.std::_Rb_tree_node_base"** %i47 to %"struct.std::_Rb_tree_node.542"**
  %i49 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i48, align 8, !tbaa !78
  %i50 = icmp eq %"struct.std::_Rb_tree_node.542"* %i49, null
  br i1 %i50, label %bb51, label %bb15, !llvm.loop !268

bb51:                                             ; preds = %bb45
  %i52 = icmp eq %"struct.std::_Rb_tree_node_base"* %i46, %i8
  br i1 %i52, label %bb76, label %bb53

bb53:                                             ; preds = %bb51
  %i54 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i46, i64 1, i32 1
  %i55 = bitcast %"struct.std::_Rb_tree_node_base"** %i54 to i64*
  %i56 = load i64, i64* %i55, align 8, !tbaa !69
  %i57 = icmp ugt i64 %i12, %i56
  %i58 = select i1 %i57, i64 %i56, i64 %i12
  %i59 = icmp eq i64 %i58, 0
  br i1 %i59, label %bb66, label %bb60

bb60:                                             ; preds = %bb53
  %i61 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i46, i64 1
  %i62 = bitcast %"struct.std::_Rb_tree_node_base"* %i61 to i8**
  %i63 = load i8*, i8** %i62, align 8, !tbaa !58
  %i64 = tail call i32 @memcmp(i8* %i14, i8* %i63, i64 %i58) #16
  %i65 = icmp eq i32 %i64, 0
  br i1 %i65, label %bb66, label %bb73

bb66:                                             ; preds = %bb60, %bb53
  %i67 = sub i64 %i12, %i56
  %i68 = icmp sgt i64 %i67, 2147483647
  br i1 %i68, label %bb83, label %bb69

bb69:                                             ; preds = %bb66
  %i70 = icmp sgt i64 %i67, -2147483648
  %i71 = select i1 %i70, i64 %i67, i64 -2147483648
  %i72 = trunc i64 %i71 to i32
  br label %bb73

bb73:                                             ; preds = %bb69, %bb60
  %i74 = phi i32 [ %i64, %bb60 ], [ %i72, %bb69 ]
  %i75 = icmp slt i32 %i74, 0
  br i1 %i75, label %bb76, label %bb83

bb76:                                             ; preds = %bb73, %bb51, %bb
  %i77 = phi %"struct.std::_Rb_tree_node_base"* [ %i46, %bb73 ], [ %i8, %bb51 ], [ %i8, %bb ]
  %i78 = getelementptr inbounds %"class.std::map", %"class.std::map"* %arg, i64 0, i32 0
  %i79 = bitcast %"class.std::tuple"* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i79) #16
  %i80 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %i, i64 0, i32 0, i32 0, i32 0
  store %"class.std::__cxx11::basic_string"* %arg1, %"class.std::__cxx11::basic_string"** %i80, align 8, !tbaa !78
  %i81 = getelementptr inbounds %"class.std::ios_base::Init", %"class.std::ios_base::Init"* %i2, i64 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %i81) #16
  %i82 = call fastcc %"struct.std::_Rb_tree_node_base"* @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %i78, %"struct.std::_Rb_tree_node_base"* %i77, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) @_ZStL19piecewise_construct.282, %"class.std::tuple"* nonnull align 8 dereferenceable(8) %i, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %i2)
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %i81) #16
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i79) #16
  br label %bb83

bb83:                                             ; preds = %bb76, %bb73, %bb66
  %i84 = phi %"struct.std::_Rb_tree_node_base"* [ %i82, %bb76 ], [ %i46, %bb73 ], [ %i46, %bb66 ]
  %i85 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i84, i64 2
  %i86 = bitcast %"struct.std::_Rb_tree_node_base"* %i85 to %"class.Kripke::Timer"*
  ret %"class.Kripke::Timer"* %i86
}

; Function Attrs: nounwind
declare i64 @_ZNSt6chrono3_V212steady_clock3nowEv() local_unnamed_addr #7

; Function Attrs: uwtable
define internal fastcc %"struct.std::_Rb_tree_node_base"* @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE22_M_emplace_hint_uniqueIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEESt17_Rb_tree_iteratorISA_ESt23_Rb_tree_const_iteratorISA_EDpOT_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node_base"* %arg1, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg2, %"class.std::tuple"* nonnull align 8 dereferenceable(8) %arg3, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg4) unnamed_addr #9 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = tail call noalias nonnull i8* @_Znwm(i64 112) #19
  %i5 = bitcast i8* %i to %"struct.std::_Rb_tree_node.542"*
  tail call fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node.542"* nonnull %i5, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg2, %"class.std::tuple"* nonnull align 8 dereferenceable(8) %arg3, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg4)
  %i6 = getelementptr inbounds i8, i8* %i, i64 32
  %i7 = bitcast i8* %i6 to %"class.std::__cxx11::basic_string"*
  %i8 = call fastcc { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node_base"* %arg1, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %i7)
  %i10 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i8, 0
  %i11 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i8, 1
  %i12 = icmp eq %"struct.std::_Rb_tree_node_base"* %i11, null
  br i1 %i12, label %bb69, label %bb13

bb13:                                             ; preds = %bb
  %i14 = icmp eq %"struct.std::_Rb_tree_node_base"* %i10, null
  br i1 %i14, label %bb16, label %bb49

bb16:                                             ; preds = %bb13
  %i17 = getelementptr inbounds %"class.std::_Rb_tree", %"class.std::_Rb_tree"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  %i18 = getelementptr inbounds i8, i8* %i17, i64 8
  %i19 = bitcast i8* %i18 to %"struct.std::_Rb_tree_node_base"*
  %i20 = icmp eq %"struct.std::_Rb_tree_node_base"* %i11, %i19
  br i1 %i20, label %bb49, label %bb21

bb21:                                             ; preds = %bb16
  %i22 = getelementptr inbounds i8, i8* %i, i64 40
  %i23 = bitcast i8* %i22 to i64*
  %i24 = load i64, i64* %i23, align 8, !tbaa !69
  %i25 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i11, i64 1, i32 1
  %i26 = bitcast %"struct.std::_Rb_tree_node_base"** %i25 to i64*
  %i27 = load i64, i64* %i26, align 8, !tbaa !69
  %i28 = icmp ugt i64 %i24, %i27
  %i29 = select i1 %i28, i64 %i27, i64 %i24
  %i30 = icmp eq i64 %i29, 0
  br i1 %i30, label %bb39, label %bb31

bb31:                                             ; preds = %bb21
  %i32 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i11, i64 1
  %i33 = bitcast %"struct.std::_Rb_tree_node_base"* %i32 to i8**
  %i34 = load i8*, i8** %i33, align 8, !tbaa !58
  %i35 = bitcast i8* %i6 to i8**
  %i36 = load i8*, i8** %i35, align 8, !tbaa !58
  %i37 = tail call i32 @memcmp(i8* %i36, i8* %i34, i64 %i29) #16
  %i38 = icmp eq i32 %i37, 0
  br i1 %i38, label %bb39, label %bb46

bb39:                                             ; preds = %bb31, %bb21
  %i40 = sub i64 %i24, %i27
  %i41 = icmp sgt i64 %i40, 2147483647
  br i1 %i41, label %bb46, label %bb42

bb42:                                             ; preds = %bb39
  %i43 = icmp sgt i64 %i40, -2147483648
  %i44 = select i1 %i43, i64 %i40, i64 -2147483648
  %i45 = trunc i64 %i44 to i32
  br label %bb46

bb46:                                             ; preds = %bb42, %bb39, %bb31
  %i47 = phi i32 [ %i37, %bb31 ], [ %i45, %bb42 ], [ 2147483647, %bb39 ]
  %i48 = icmp slt i32 %i47, 0
  br label %bb49

bb49:                                             ; preds = %bb46, %bb16, %bb13
  %i50 = phi i1 [ true, %bb16 ], [ %i48, %bb46 ], [ true, %bb13 ]
  %i51 = bitcast i8* %i to %"struct.std::_Rb_tree_node_base"*
  %i52 = getelementptr inbounds %"class.std::_Rb_tree", %"class.std::_Rb_tree"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  %i53 = getelementptr inbounds i8, i8* %i52, i64 8
  %i54 = bitcast i8* %i53 to %"struct.std::_Rb_tree_node_base"*
  tail call void @_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_node_baseS0_RS_(i1 zeroext %i50, %"struct.std::_Rb_tree_node_base"* nonnull %i51, %"struct.std::_Rb_tree_node_base"* nonnull %i11, %"struct.std::_Rb_tree_node_base"* nonnull align 8 dereferenceable(32) %i54) #16
  %i55 = getelementptr inbounds i8, i8* %i52, i64 40
  %i56 = bitcast i8* %i55 to i64*
  %i57 = load i64, i64* %i56, align 8, !tbaa !269
  %i58 = add i64 %i57, 1
  store i64 %i58, i64* %i56, align 8, !tbaa !269
  br label %bb76

bb69:                                             ; preds = %bb
  %i70 = bitcast i8* %i6 to i8**
  %i71 = load i8*, i8** %i70, align 8, !tbaa !58
  %i72 = getelementptr inbounds i8, i8* %i, i64 48
  %i73 = icmp eq i8* %i71, %i72
  br i1 %i73, label %bb75, label %bb74

bb74:                                             ; preds = %bb69
  tail call void @_ZdlPv(i8* %i71) #16
  br label %bb75

bb75:                                             ; preds = %bb74, %bb69
  tail call void @_ZdlPv(i8* nonnull %i) #16
  br label %bb76

bb76:                                             ; preds = %bb75, %bb49
  %i77 = phi %"struct.std::_Rb_tree_node_base"* [ %i10, %bb75 ], [ %i51, %bb49 ]
  ret %"struct.std::_Rb_tree_node_base"* %i77
}

; Function Attrs: uwtable
define internal fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE17_M_construct_nodeIJRKSt21piecewise_construct_tSt5tupleIJRS7_EESL_IJEEEEEvPSt13_Rb_tree_nodeISA_EDpOT_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node.542"* %arg1, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg2, %"class.std::tuple"* nonnull align 8 dereferenceable(8) %arg3, %"class.std::ios_base::Init"* nonnull align 1 dereferenceable(1) %arg4) unnamed_addr #9 comdat align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = alloca i64, align 8
  %i5 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1
  %i6 = getelementptr inbounds %"class.std::tuple", %"class.std::tuple"* %arg3, i64 0, i32 0, i32 0, i32 0
  %i7 = load %"class.std::__cxx11::basic_string"*, %"class.std::__cxx11::basic_string"** %i6, align 8, !tbaa !270
  %i8 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1, i32 0, i64 16
  %i9 = bitcast %"struct.__gnu_cxx::__aligned_membuf.541"* %i5 to i8**
  store i8* %i8, i8** %i9, align 8, !tbaa !67
  %i10 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i7, i64 0, i32 0, i32 0
  %i11 = load i8*, i8** %i10, align 8, !tbaa !58
  %i12 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %i7, i64 0, i32 1
  %i13 = load i64, i64* %i12, align 8, !tbaa !69
  %i14 = bitcast i64* %i to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i14) #16
  store i64 %i13, i64* %i, align 8, !tbaa !191
  %i15 = icmp ugt i64 %i13, 15
  br i1 %i15, label %bb16, label %bb22

bb16:                                             ; preds = %bb
  %i17 = bitcast %"struct.__gnu_cxx::__aligned_membuf.541"* %i5 to %"class.std::__cxx11::basic_string"*
  %i18 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %i17, i64* nonnull align 8 dereferenceable(8) %i, i64 0)
  store i8* %i18, i8** %i9, align 8, !tbaa !58
  %i20 = load i64, i64* %i, align 8, !tbaa !191
  %i21 = bitcast i8* %i8 to i64*
  store i64 %i20, i64* %i21, align 8, !tbaa !68
  br label %bb22

bb22:                                             ; preds = %bb16, %bb
  %i23 = phi i8* [ %i18, %bb16 ], [ %i8, %bb ]
  switch i64 %i13, label %bb26 [
    i64 1, label %bb24
    i64 0, label %bb34
  ]

bb24:                                             ; preds = %bb22
  %i25 = load i8, i8* %i11, align 1, !tbaa !68
  store i8 %i25, i8* %i23, align 1, !tbaa !68
  br label %bb34

bb26:                                             ; preds = %bb22
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %i23, i8* align 1 %i11, i64 %i13, i1 false) #16
  br label %bb34

bb34:                                             ; preds = %bb26, %bb24, %bb22
  %i35 = load i64, i64* %i, align 8, !tbaa !191
  %i36 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1, i32 0, i64 8
  %i37 = bitcast i8* %i36 to i64*
  store i64 %i35, i64* %i37, align 8, !tbaa !69
  %i38 = load i8*, i8** %i9, align 8, !tbaa !58
  %i39 = getelementptr inbounds i8, i8* %i38, i64 %i35
  store i8 0, i8* %i39, align 1, !tbaa !68
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i14) #16
  %i40 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1, i32 0, i64 32
  store i8 0, i8* %i40, align 8, !tbaa !263
  %i41 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1, i32 0, i64 40
  call void @llvm.memset.p0i8.i64(i8* nonnull align 8 dereferenceable(16) %i41, i8 0, i64 16, i1 false)
  %i42 = call i64 @_ZNSt6chrono3_V212steady_clock3nowEv() #16
  %i43 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1, i32 0, i64 56
  %i44 = bitcast i8* %i43 to i64*
  store i64 %i42, i64* %i44, align 8
  %i45 = call i64 @_ZNSt6chrono3_V212steady_clock3nowEv() #16
  %i46 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1, i32 0, i64 64
  %i47 = bitcast i8* %i46 to i64*
  store i64 %i45, i64* %i47, align 8
  %i48 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %arg1, i64 0, i32 1, i32 0, i64 72
  %i49 = bitcast i8* %i48 to double*
  store double 0.000000e+00, double* %i49, align 8, !tbaa !259
  ret void
}

; Function Attrs: uwtable
define internal fastcc { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE29_M_get_insert_hint_unique_posESt23_Rb_tree_const_iteratorISA_ERS7_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node_base"* %arg1, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg2) unnamed_addr #9 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.std::_Rb_tree", %"class.std::_Rb_tree"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  %i3 = getelementptr inbounds i8, i8* %i, i64 8
  %i4 = bitcast i8* %i3 to %"struct.std::_Rb_tree_node_base"*
  %i5 = icmp eq %"struct.std::_Rb_tree_node_base"* %i4, %arg1
  br i1 %i5, label %bb6, label %bb45

bb6:                                              ; preds = %bb
  %i7 = getelementptr inbounds i8, i8* %i, i64 40
  %i8 = bitcast i8* %i7 to i64*
  %i9 = load i64, i64* %i8, align 8, !tbaa !269
  %i10 = icmp eq i64 %i9, 0
  br i1 %i10, label %bb41, label %bb11

bb11:                                             ; preds = %bb6
  %i12 = getelementptr inbounds i8, i8* %i, i64 32
  %i13 = bitcast i8* %i12 to %"struct.std::_Rb_tree_node_base"**
  %i14 = load %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"** %i13, align 8, !tbaa !78
  %i15 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i14, i64 1, i32 1
  %i16 = bitcast %"struct.std::_Rb_tree_node_base"** %i15 to i64*
  %i17 = load i64, i64* %i16, align 8, !tbaa !69
  %i18 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 1
  %i19 = load i64, i64* %i18, align 8, !tbaa !69
  %i20 = icmp ugt i64 %i17, %i19
  %i21 = select i1 %i20, i64 %i19, i64 %i17
  %i22 = icmp eq i64 %i21, 0
  br i1 %i22, label %bb31, label %bb23

bb23:                                             ; preds = %bb11
  %i24 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i14, i64 1
  %i25 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i26 = load i8*, i8** %i25, align 8, !tbaa !58
  %i27 = bitcast %"struct.std::_Rb_tree_node_base"* %i24 to i8**
  %i28 = load i8*, i8** %i27, align 8, !tbaa !58
  %i29 = tail call i32 @memcmp(i8* %i28, i8* %i26, i64 %i21) #16
  %i30 = icmp eq i32 %i29, 0
  br i1 %i30, label %bb31, label %bb38

bb31:                                             ; preds = %bb23, %bb11
  %i32 = sub i64 %i17, %i19
  %i33 = icmp sgt i64 %i32, 2147483647
  br i1 %i33, label %bb41, label %bb34

bb34:                                             ; preds = %bb31
  %i35 = icmp sgt i64 %i32, -2147483648
  %i36 = select i1 %i35, i64 %i32, i64 -2147483648
  %i37 = trunc i64 %i36 to i32
  br label %bb38

bb38:                                             ; preds = %bb34, %bb23
  %i39 = phi i32 [ %i29, %bb23 ], [ %i37, %bb34 ]
  %i40 = icmp slt i32 %i39, 0
  br i1 %i40, label %bb174, label %bb41

bb41:                                             ; preds = %bb38, %bb31, %bb6
  %i42 = tail call fastcc { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg2)
  %i43 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i42, 0
  %i44 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i42, 1
  br label %bb174

bb45:                                             ; preds = %bb
  %i46 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %arg1, i64 1
  %i47 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 1
  %i48 = load i64, i64* %i47, align 8, !tbaa !69
  %i49 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %arg1, i64 1, i32 1
  %i50 = bitcast %"struct.std::_Rb_tree_node_base"** %i49 to i64*
  %i51 = load i64, i64* %i50, align 8, !tbaa !69
  %i52 = icmp ugt i64 %i48, %i51
  %i53 = select i1 %i52, i64 %i51, i64 %i48
  %i54 = icmp eq i64 %i53, 0
  br i1 %i54, label %bb62, label %bb55

bb55:                                             ; preds = %bb45
  %i56 = bitcast %"struct.std::_Rb_tree_node_base"* %i46 to i8**
  %i57 = load i8*, i8** %i56, align 8, !tbaa !58
  %i58 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i59 = load i8*, i8** %i58, align 8, !tbaa !58
  %i60 = tail call i32 @memcmp(i8* %i59, i8* %i57, i64 %i53) #16
  %i61 = icmp eq i32 %i60, 0
  br i1 %i61, label %bb62, label %bb69

bb62:                                             ; preds = %bb55, %bb45
  %i63 = sub i64 %i48, %i51
  %i64 = icmp sgt i64 %i63, 2147483647
  br i1 %i64, label %bb114, label %bb65

bb65:                                             ; preds = %bb62
  %i66 = icmp sgt i64 %i63, -2147483648
  %i67 = select i1 %i66, i64 %i63, i64 -2147483648
  %i68 = trunc i64 %i67 to i32
  br label %bb69

bb69:                                             ; preds = %bb65, %bb55
  %i70 = phi i32 [ %i60, %bb55 ], [ %i68, %bb65 ]
  %i71 = icmp slt i32 %i70, 0
  br i1 %i71, label %bb72, label %bb114

bb72:                                             ; preds = %bb69
  %i73 = getelementptr inbounds i8, i8* %i, i64 24
  %i74 = bitcast i8* %i73 to %"struct.std::_Rb_tree_node_base"**
  %i75 = load %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"** %i74, align 8, !tbaa !78
  %i76 = icmp eq %"struct.std::_Rb_tree_node_base"* %i75, %arg1
  br i1 %i76, label %bb174, label %bb77

bb77:                                             ; preds = %bb72
  %i78 = tail call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* nonnull %arg1) #20
  %i79 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i78, i64 1, i32 1
  %i80 = bitcast %"struct.std::_Rb_tree_node_base"** %i79 to i64*
  %i81 = load i64, i64* %i80, align 8, !tbaa !69
  %i82 = icmp ugt i64 %i81, %i48
  %i83 = select i1 %i82, i64 %i48, i64 %i81
  %i84 = icmp eq i64 %i83, 0
  br i1 %i84, label %bb93, label %bb85

bb85:                                             ; preds = %bb77
  %i86 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i78, i64 1
  %i87 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i88 = load i8*, i8** %i87, align 8, !tbaa !58
  %i89 = bitcast %"struct.std::_Rb_tree_node_base"* %i86 to i8**
  %i90 = load i8*, i8** %i89, align 8, !tbaa !58
  %i91 = tail call i32 @memcmp(i8* %i90, i8* %i88, i64 %i83) #16
  %i92 = icmp eq i32 %i91, 0
  br i1 %i92, label %bb93, label %bb100

bb93:                                             ; preds = %bb85, %bb77
  %i94 = sub i64 %i81, %i48
  %i95 = icmp sgt i64 %i94, 2147483647
  br i1 %i95, label %bb110, label %bb96

bb96:                                             ; preds = %bb93
  %i97 = icmp sgt i64 %i94, -2147483648
  %i98 = select i1 %i97, i64 %i94, i64 -2147483648
  %i99 = trunc i64 %i98 to i32
  br label %bb100

bb100:                                            ; preds = %bb96, %bb85
  %i101 = phi i32 [ %i91, %bb85 ], [ %i99, %bb96 ]
  %i102 = icmp slt i32 %i101, 0
  br i1 %i102, label %bb103, label %bb110

bb103:                                            ; preds = %bb100
  %i104 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i78, i64 0, i32 3
  %i105 = bitcast %"struct.std::_Rb_tree_node_base"** %i104 to %"struct.std::_Rb_tree_node.542"**
  %i106 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i105, align 8, !tbaa !51
  %i107 = icmp eq %"struct.std::_Rb_tree_node.542"* %i106, null
  %i108 = select i1 %i107, %"struct.std::_Rb_tree_node_base"* null, %"struct.std::_Rb_tree_node_base"* %arg1
  %i109 = select i1 %i107, %"struct.std::_Rb_tree_node_base"* %i78, %"struct.std::_Rb_tree_node_base"* %arg1
  br label %bb174

bb110:                                            ; preds = %bb100, %bb93
  %i111 = tail call fastcc { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg2)
  %i112 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i111, 0
  %i113 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i111, 1
  br label %bb174

bb114:                                            ; preds = %bb69, %bb62
  br i1 %i54, label %bb122, label %bb115

bb115:                                            ; preds = %bb114
  %i116 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i117 = load i8*, i8** %i116, align 8, !tbaa !58
  %i118 = bitcast %"struct.std::_Rb_tree_node_base"* %i46 to i8**
  %i119 = load i8*, i8** %i118, align 8, !tbaa !58
  %i120 = tail call i32 @memcmp(i8* %i119, i8* %i117, i64 %i53) #16
  %i121 = icmp eq i32 %i120, 0
  br i1 %i121, label %bb122, label %bb129

bb122:                                            ; preds = %bb115, %bb114
  %i123 = sub i64 %i51, %i48
  %i124 = icmp sgt i64 %i123, 2147483647
  br i1 %i124, label %bb174, label %bb125

bb125:                                            ; preds = %bb122
  %i126 = icmp sgt i64 %i123, -2147483648
  %i127 = select i1 %i126, i64 %i123, i64 -2147483648
  %i128 = trunc i64 %i127 to i32
  br label %bb129

bb129:                                            ; preds = %bb125, %bb115
  %i130 = phi i32 [ %i120, %bb115 ], [ %i128, %bb125 ]
  %i131 = icmp slt i32 %i130, 0
  br i1 %i131, label %bb132, label %bb174

bb132:                                            ; preds = %bb129
  %i133 = getelementptr inbounds i8, i8* %i, i64 32
  %i134 = bitcast i8* %i133 to %"struct.std::_Rb_tree_node_base"**
  %i135 = load %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"** %i134, align 8, !tbaa !78
  %i136 = icmp eq %"struct.std::_Rb_tree_node_base"* %i135, %arg1
  br i1 %i136, label %bb174, label %bb137

bb137:                                            ; preds = %bb132
  %i138 = tail call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* nonnull %arg1) #20
  %i139 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i138, i64 1, i32 1
  %i140 = bitcast %"struct.std::_Rb_tree_node_base"** %i139 to i64*
  %i141 = load i64, i64* %i140, align 8, !tbaa !69
  %i142 = icmp ugt i64 %i48, %i141
  %i143 = select i1 %i142, i64 %i141, i64 %i48
  %i144 = icmp eq i64 %i143, 0
  br i1 %i144, label %bb153, label %bb145

bb145:                                            ; preds = %bb137
  %i146 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i138, i64 1
  %i147 = bitcast %"struct.std::_Rb_tree_node_base"* %i146 to i8**
  %i148 = load i8*, i8** %i147, align 8, !tbaa !58
  %i149 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg2, i64 0, i32 0, i32 0
  %i150 = load i8*, i8** %i149, align 8, !tbaa !58
  %i151 = tail call i32 @memcmp(i8* %i150, i8* %i148, i64 %i143) #16
  %i152 = icmp eq i32 %i151, 0
  br i1 %i152, label %bb153, label %bb160

bb153:                                            ; preds = %bb145, %bb137
  %i154 = sub i64 %i48, %i141
  %i155 = icmp sgt i64 %i154, 2147483647
  br i1 %i155, label %bb170, label %bb156

bb156:                                            ; preds = %bb153
  %i157 = icmp sgt i64 %i154, -2147483648
  %i158 = select i1 %i157, i64 %i154, i64 -2147483648
  %i159 = trunc i64 %i158 to i32
  br label %bb160

bb160:                                            ; preds = %bb156, %bb145
  %i161 = phi i32 [ %i151, %bb145 ], [ %i159, %bb156 ]
  %i162 = icmp slt i32 %i161, 0
  br i1 %i162, label %bb163, label %bb170

bb163:                                            ; preds = %bb160
  %i164 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %arg1, i64 0, i32 3
  %i165 = bitcast %"struct.std::_Rb_tree_node_base"** %i164 to %"struct.std::_Rb_tree_node.542"**
  %i166 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i165, align 8, !tbaa !51
  %i167 = icmp eq %"struct.std::_Rb_tree_node.542"* %i166, null
  %i168 = select i1 %i167, %"struct.std::_Rb_tree_node_base"* null, %"struct.std::_Rb_tree_node_base"* %i138
  %i169 = select i1 %i167, %"struct.std::_Rb_tree_node_base"* %arg1, %"struct.std::_Rb_tree_node_base"* %i138
  br label %bb174

bb170:                                            ; preds = %bb160, %bb153
  %i171 = tail call fastcc { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg2)
  %i172 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i171, 0
  %i173 = extractvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i171, 1
  br label %bb174

bb174:                                            ; preds = %bb170, %bb163, %bb132, %bb129, %bb122, %bb110, %bb103, %bb72, %bb41, %bb38
  %i175 = phi %"struct.std::_Rb_tree_node_base"* [ %i43, %bb41 ], [ null, %bb38 ], [ %i112, %bb110 ], [ %arg1, %bb72 ], [ %i172, %bb170 ], [ null, %bb132 ], [ %arg1, %bb129 ], [ %arg1, %bb122 ], [ %i108, %bb103 ], [ %i168, %bb163 ]
  %i176 = phi %"struct.std::_Rb_tree_node_base"* [ %i44, %bb41 ], [ %i14, %bb38 ], [ %i113, %bb110 ], [ %arg1, %bb72 ], [ %i173, %bb170 ], [ %arg1, %bb132 ], [ null, %bb129 ], [ null, %bb122 ], [ %i109, %bb103 ], [ %i169, %bb163 ]
  %i177 = insertvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } undef, %"struct.std::_Rb_tree_node_base"* %i175, 0
  %i178 = insertvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i177, %"struct.std::_Rb_tree_node_base"* %i176, 1
  ret { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i178
}

; Function Attrs: uwtable
define internal fastcc { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE24_M_get_insert_unique_posERS7_(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %arg1) unnamed_addr #9 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.std::_Rb_tree", %"class.std::_Rb_tree"* %arg, i64 0, i32 0, i32 0, i32 0, i32 0
  %i2 = getelementptr inbounds i8, i8* %i, i64 16
  %i3 = bitcast i8* %i2 to %"struct.std::_Rb_tree_node.542"**
  %i4 = getelementptr inbounds i8, i8* %i, i64 8
  %i5 = bitcast i8* %i4 to %"struct.std::_Rb_tree_node_base"*
  %i6 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i3, align 8, !tbaa !78
  %i7 = icmp eq %"struct.std::_Rb_tree_node.542"* %i6, null
  br i1 %i7, label %bb53, label %bb8

bb8:                                              ; preds = %bb
  %i9 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 1
  %i10 = load i64, i64* %i9, align 8, !tbaa !69
  %i11 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 0, i32 0
  %i12 = load i8*, i8** %i11, align 8
  br label %bb13

bb13:                                             ; preds = %bb42, %bb8
  %i14 = phi %"struct.std::_Rb_tree_node.542"* [ %i6, %bb8 ], [ %i43, %bb42 ]
  %i15 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i14, i64 0, i32 1, i32 0, i64 8
  %i16 = bitcast i8* %i15 to i64*
  %i17 = load i64, i64* %i16, align 8, !tbaa !69
  %i18 = icmp ugt i64 %i10, %i17
  %i19 = select i1 %i18, i64 %i17, i64 %i10
  %i20 = icmp eq i64 %i19, 0
  br i1 %i20, label %bb27, label %bb21

bb21:                                             ; preds = %bb13
  %i22 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i14, i64 0, i32 1
  %i23 = bitcast %"struct.__gnu_cxx::__aligned_membuf.541"* %i22 to i8**
  %i24 = load i8*, i8** %i23, align 8, !tbaa !58
  %i25 = tail call i32 @memcmp(i8* %i12, i8* %i24, i64 %i19) #16
  %i26 = icmp eq i32 %i25, 0
  br i1 %i26, label %bb27, label %bb34

bb27:                                             ; preds = %bb21, %bb13
  %i28 = sub i64 %i10, %i17
  %i29 = icmp sgt i64 %i28, 2147483647
  br i1 %i29, label %bb44, label %bb30

bb30:                                             ; preds = %bb27
  %i31 = icmp sgt i64 %i28, -2147483648
  %i32 = select i1 %i31, i64 %i28, i64 -2147483648
  %i33 = trunc i64 %i32 to i32
  br label %bb34

bb34:                                             ; preds = %bb30, %bb21
  %i35 = phi i32 [ %i25, %bb21 ], [ %i33, %bb30 ]
  %i36 = icmp slt i32 %i35, 0
  br i1 %i36, label %bb37, label %bb44

bb37:                                             ; preds = %bb34
  %i38 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i14, i64 0, i32 0, i32 2
  %i39 = bitcast %"struct.std::_Rb_tree_node_base"** %i38 to %"struct.std::_Rb_tree_node.542"**
  %i40 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i39, align 8, !tbaa !78
  %i41 = icmp eq %"struct.std::_Rb_tree_node.542"* %i40, null
  br i1 %i41, label %bb51, label %bb42

bb42:                                             ; preds = %bb44, %bb37
  %i43 = phi %"struct.std::_Rb_tree_node.542"* [ %i40, %bb37 ], [ %i47, %bb44 ]
  br label %bb13, !llvm.loop !272

bb44:                                             ; preds = %bb34, %bb27
  %i45 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i14, i64 0, i32 0, i32 3
  %i46 = bitcast %"struct.std::_Rb_tree_node_base"** %i45 to %"struct.std::_Rb_tree_node.542"**
  %i47 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i46, align 8, !tbaa !78
  %i48 = icmp eq %"struct.std::_Rb_tree_node.542"* %i47, null
  br i1 %i48, label %bb49, label %bb42

bb49:                                             ; preds = %bb44
  %i50 = getelementptr %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i14, i64 0, i32 0
  br label %bb63

bb51:                                             ; preds = %bb37
  %i52 = getelementptr %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i14, i64 0, i32 0
  br label %bb53

bb53:                                             ; preds = %bb51, %bb
  %i54 = phi %"struct.std::_Rb_tree_node_base"* [ %i52, %bb51 ], [ %i5, %bb ]
  %i55 = getelementptr inbounds i8, i8* %i, i64 24
  %i56 = bitcast i8* %i55 to %"struct.std::_Rb_tree_node_base"**
  %i57 = load %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"** %i56, align 8, !tbaa !273
  %i58 = icmp eq %"struct.std::_Rb_tree_node_base"* %i54, %i57
  br i1 %i58, label %bb94, label %bb59

bb59:                                             ; preds = %bb53
  %i60 = tail call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* nonnull %i54) #20
  %i61 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 1
  %i62 = load i64, i64* %i61, align 8, !tbaa !69
  br label %bb63

bb63:                                             ; preds = %bb59, %bb49
  %i64 = phi i64 [ %i62, %bb59 ], [ %i10, %bb49 ]
  %i65 = phi %"struct.std::_Rb_tree_node_base"* [ %i54, %bb59 ], [ %i50, %bb49 ]
  %i66 = phi %"struct.std::_Rb_tree_node_base"* [ %i60, %bb59 ], [ %i50, %bb49 ]
  %i67 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i66, i64 1, i32 1
  %i68 = bitcast %"struct.std::_Rb_tree_node_base"** %i67 to i64*
  %i69 = load i64, i64* %i68, align 8, !tbaa !69
  %i71 = icmp ugt i64 %i69, %i64
  %i72 = select i1 %i71, i64 %i64, i64 %i69
  %i73 = icmp eq i64 %i72, 0
  br i1 %i73, label %bb82, label %bb74

bb74:                                             ; preds = %bb63
  %i75 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i66, i64 1
  %i76 = getelementptr inbounds %"class.std::__cxx11::basic_string", %"class.std::__cxx11::basic_string"* %arg1, i64 0, i32 0, i32 0
  %i77 = load i8*, i8** %i76, align 8, !tbaa !58
  %i78 = bitcast %"struct.std::_Rb_tree_node_base"* %i75 to i8**
  %i79 = load i8*, i8** %i78, align 8, !tbaa !58
  %i80 = tail call i32 @memcmp(i8* %i79, i8* %i77, i64 %i72) #16
  %i81 = icmp eq i32 %i80, 0
  br i1 %i81, label %bb82, label %bb89

bb82:                                             ; preds = %bb74, %bb63
  %i83 = sub i64 %i69, %i64
  %i84 = icmp sgt i64 %i83, 2147483647
  br i1 %i84, label %bb94, label %bb85

bb85:                                             ; preds = %bb82
  %i86 = icmp sgt i64 %i83, -2147483648
  %i87 = select i1 %i86, i64 %i83, i64 -2147483648
  %i88 = trunc i64 %i87 to i32
  br label %bb89

bb89:                                             ; preds = %bb85, %bb74
  %i90 = phi i32 [ %i80, %bb74 ], [ %i88, %bb85 ]
  %i91 = icmp slt i32 %i90, 0
  %i92 = select i1 %i91, %"struct.std::_Rb_tree_node_base"* null, %"struct.std::_Rb_tree_node_base"* %i66
  %i93 = select i1 %i91, %"struct.std::_Rb_tree_node_base"* %i65, %"struct.std::_Rb_tree_node_base"* null
  br label %bb94

bb94:                                             ; preds = %bb89, %bb82, %bb53
  %i95 = phi %"struct.std::_Rb_tree_node_base"* [ %i66, %bb82 ], [ %i92, %bb89 ], [ null, %bb53 ]
  %i96 = phi %"struct.std::_Rb_tree_node_base"* [ null, %bb82 ], [ %i93, %bb89 ], [ %i54, %bb53 ]
  %i97 = insertvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } undef, %"struct.std::_Rb_tree_node_base"* %i95, 0
  %i98 = insertvalue { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i97, %"struct.std::_Rb_tree_node_base"* %i96, 1
  ret { %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* } %i98
}

; Function Attrs: uwtable
define internal fastcc void @_ZN6Kripke6Timing7stopAllEv(%"class.Kripke::Timing"* nonnull dereferenceable(64) %arg) unnamed_addr #9 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = alloca i64, align 8
  %i1 = alloca %"struct.std::pair", align 8
  %i2 = getelementptr inbounds %"class.Kripke::Timing", %"class.Kripke::Timing"* %arg, i64 0, i32 1
  %i3 = getelementptr inbounds %"class.std::map", %"class.std::map"* %i2, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %i4 = getelementptr inbounds i8, i8* %i3, i64 24
  %i5 = bitcast i8* %i4 to %"struct.std::_Rb_tree_node_base"**
  %i6 = load %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"** %i5, align 8, !tbaa !273
  %i7 = getelementptr inbounds i8, i8* %i3, i64 8
  %i8 = bitcast i8* %i7 to %"struct.std::_Rb_tree_node_base"*
  %i9 = icmp eq %"struct.std::_Rb_tree_node_base"* %i6, %i8
  br i1 %i9, label %bb21, label %bb10

bb10:                                             ; preds = %bb
  %i11 = bitcast %"struct.std::pair"* %i1 to i8*
  %i12 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %i1, i64 0, i32 0, i32 2
  %i13 = bitcast %"struct.std::pair"* %i1 to %union.anon**
  %i14 = bitcast i64* %i to i8*
  %i15 = bitcast %union.anon* %i12 to i8*
  %i16 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %i1, i64 0, i32 0
  %i17 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %i1, i64 0, i32 0, i32 0, i32 0
  %i18 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %i1, i64 0, i32 0, i32 2, i32 0
  %i19 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %i1, i64 0, i32 0, i32 1
  %i20 = getelementptr inbounds %"struct.std::pair", %"struct.std::pair"* %i1, i64 0, i32 1, i32 0
  br label %bb22

bb21:                                             ; preds = %bb68, %bb
  ret void

bb22:                                             ; preds = %bb68, %bb10
  %i23 = phi %"struct.std::_Rb_tree_node_base"* [ %i6, %bb10 ], [ %i69, %bb68 ]
  call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %i11) #16
  %i24 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i23, i64 1
  store %union.anon* %i12, %union.anon** %i13, align 8, !tbaa !67
  %i25 = bitcast %"struct.std::_Rb_tree_node_base"* %i24 to i8**
  %i26 = load i8*, i8** %i25, align 8, !tbaa !58
  %i27 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i23, i64 1, i32 1
  %i28 = bitcast %"struct.std::_Rb_tree_node_base"** %i27 to i64*
  %i29 = load i64, i64* %i28, align 8, !tbaa !69
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %i14) #16
  store i64 %i29, i64* %i, align 8, !tbaa !191
  %i30 = icmp ugt i64 %i29, 15
  br i1 %i30, label %bb31, label %bb34

bb31:                                             ; preds = %bb22
  %i32 = call i8* @_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm(%"class.std::__cxx11::basic_string"* nonnull dereferenceable(32) %i16, i64* nonnull align 8 dereferenceable(8) %i, i64 0)
  store i8* %i32, i8** %i17, align 8, !tbaa !58
  %i33 = load i64, i64* %i, align 8, !tbaa !191
  store i64 %i33, i64* %i18, align 8, !tbaa !68
  br label %bb34

bb34:                                             ; preds = %bb31, %bb22
  %i35 = phi i8* [ %i32, %bb31 ], [ %i15, %bb22 ]
  switch i64 %i29, label %bb38 [
    i64 1, label %bb36
    i64 0, label %bb39
  ]

bb36:                                             ; preds = %bb34
  %i37 = load i8, i8* %i26, align 1, !tbaa !68
  store i8 %i37, i8* %i35, align 1, !tbaa !68
  br label %bb39

bb38:                                             ; preds = %bb34
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %i35, i8* align 1 %i26, i64 %i29, i1 false) #16
  br label %bb39

bb39:                                             ; preds = %bb38, %bb36, %bb34
  %i40 = load i64, i64* %i, align 8, !tbaa !191
  store i64 %i40, i64* %i19, align 8, !tbaa !69
  %i41 = load i8*, i8** %i17, align 8, !tbaa !58
  %i42 = getelementptr inbounds i8, i8* %i41, i64 %i40
  store i8 0, i8* %i42, align 1, !tbaa !68
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %i14) #16
  %i43 = getelementptr inbounds %"struct.std::_Rb_tree_node_base", %"struct.std::_Rb_tree_node_base"* %i23, i64 2
  %i44 = bitcast %"struct.std::_Rb_tree_node_base"* %i43 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 8 dereferenceable(48) %i20, i8* nonnull align 8 dereferenceable(48) %i44, i64 48, i1 false)
  %i45 = call fastcc nonnull align 8 dereferenceable(48) %"class.Kripke::Timer"* @_ZNSt3mapINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEN6Kripke5TimerESt4lessIS5_ESaISt4pairIKS5_S7_EEEixERSB_(%"class.std::map"* nonnull dereferenceable(48) %i2, %"class.std::__cxx11::basic_string"* nonnull align 8 dereferenceable(32) %i16)
  %i47 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i45, i64 0, i32 0
  %i48 = load i8, i8* %i47, align 8, !tbaa !263, !range !274
  %i49 = icmp eq i8 %i48, 0
  br i1 %i49, label %bb64, label %bb50

bb50:                                             ; preds = %bb39
  %i51 = call i64 @_ZNSt6chrono3_V212steady_clock3nowEv() #16
  %i52 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i45, i64 0, i32 3, i32 0, i32 1, i32 0, i32 0
  store i64 %i51, i64* %i52, align 8, !tbaa.struct !258
  %i53 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i45, i64 0, i32 3, i32 0, i32 0, i32 0, i32 0
  %i54 = load i64, i64* %i53, align 8, !tbaa.struct !258
  %i55 = sub nsw i64 %i51, %i54
  %i56 = sitofp i64 %i55 to double
  %i57 = fmul fast double %i56, 1.000000e-09
  %i58 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i45, i64 0, i32 3, i32 0, i32 2
  %i59 = load double, double* %i58, align 8, !tbaa !259
  %i60 = fadd fast double %i57, %i59
  store double %i60, double* %i58, align 8, !tbaa !259
  %i61 = getelementptr inbounds %"class.Kripke::Timer", %"class.Kripke::Timer"* %i45, i64 0, i32 1
  %i62 = load double, double* %i61, align 8, !tbaa !275
  %i63 = fadd fast double %i60, %i62
  store double %i63, double* %i61, align 8, !tbaa !275
  br label %bb64

bb64:                                             ; preds = %bb50, %bb39
  %i65 = load i8*, i8** %i17, align 8, !tbaa !58
  %i66 = icmp eq i8* %i65, %i15
  br i1 %i66, label %bb68, label %bb67

bb67:                                             ; preds = %bb64
  call void @_ZdlPv(i8* %i65) #16
  br label %bb68

bb68:                                             ; preds = %bb67, %bb64
  call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %i11) #16
  %i69 = call %"struct.std::_Rb_tree_node_base"* @_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base(%"struct.std::_Rb_tree_node_base"* nonnull %i23) #20
  %i70 = icmp eq %"struct.std::_Rb_tree_node_base"* %i69, %i8
  br i1 %i70, label %bb21, label %bb22, !llvm.loop !276
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke6TimingD0Ev(%"class.Kripke::Timing"* nonnull dereferenceable(64) %arg) unnamed_addr #12 align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = getelementptr inbounds %"class.Kripke::Timing", %"class.Kripke::Timing"* %arg, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke6TimingE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  call fastcc void @_ZN6Kripke6Timing7stopAllEv(%"class.Kripke::Timing"* nonnull dereferenceable(64) %arg) #16
  %i2 = getelementptr inbounds %"class.Kripke::Timing", %"class.Kripke::Timing"* %arg, i64 0, i32 1
  %i3 = getelementptr inbounds %"class.std::map", %"class.std::map"* %i2, i64 0, i32 0
  %i4 = getelementptr inbounds %"class.std::map", %"class.std::map"* %i2, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %i5 = getelementptr inbounds i8, i8* %i4, i64 16
  %i6 = bitcast i8* %i5 to %"struct.std::_Rb_tree_node.542"**
  %i7 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i6, align 8, !tbaa !192
  call fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %i3, %"struct.std::_Rb_tree_node.542"* %i7) #16
  %i16 = bitcast %"class.Kripke::Timing"* %arg to i8*
  tail call void @_ZdlPv(i8* nonnull %i16) #17
  ret void
}

; Function Attrs: uwtable
define internal fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node.542"* %arg1) unnamed_addr #9 comdat align 2 personality i32 (...)* @__gxx_personality_v0 {
bb:
  %i = icmp eq %"struct.std::_Rb_tree_node.542"* %arg1, null
  br i1 %i, label %bb19, label %bb2

bb2:                                              ; preds = %bb16, %bb
  %i3 = phi %"struct.std::_Rb_tree_node.542"* [ %i9, %bb16 ], [ %arg1, %bb ]
  %i4 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i3, i64 0, i32 0, i32 3
  %i5 = bitcast %"struct.std::_Rb_tree_node_base"** %i4 to %"struct.std::_Rb_tree_node.542"**
  %i6 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i5, align 8, !tbaa !51
  tail call fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %arg, %"struct.std::_Rb_tree_node.542"* %i6)
  %i7 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i3, i64 0, i32 0, i32 2
  %i8 = bitcast %"struct.std::_Rb_tree_node_base"** %i7 to %"struct.std::_Rb_tree_node.542"**
  %i9 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i8, align 8, !tbaa !57
  %i10 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i3, i64 0, i32 1
  %i11 = bitcast %"struct.__gnu_cxx::__aligned_membuf.541"* %i10 to i8**
  %i12 = load i8*, i8** %i11, align 8, !tbaa !58
  %i13 = getelementptr inbounds %"struct.std::_Rb_tree_node.542", %"struct.std::_Rb_tree_node.542"* %i3, i64 0, i32 1, i32 0, i64 16
  %i14 = icmp eq i8* %i12, %i13
  br i1 %i14, label %bb16, label %bb15

bb15:                                             ; preds = %bb2
  tail call void @_ZdlPv(i8* %i12) #16
  br label %bb16

bb16:                                             ; preds = %bb15, %bb2
  %i17 = bitcast %"struct.std::_Rb_tree_node.542"* %i3 to i8*
  tail call void @_ZdlPv(i8* nonnull %i17) #16
  %i18 = icmp eq %"struct.std::_Rb_tree_node.542"* %i9, null
  br i1 %i18, label %bb19, label %bb2, !llvm.loop !277

bb19:                                             ; preds = %bb16, %bb
  ret void
}

; Function Attrs: nounwind uwtable
define internal void @_ZN6Kripke6TimingD2Ev(%"class.Kripke::Timing"* nonnull dereferenceable(64) %arg) unnamed_addr #12 align 2 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %i = getelementptr inbounds %"class.Kripke::Timing", %"class.Kripke::Timing"* %arg, i64 0, i32 0, i32 0
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [4 x i8*] }, { [4 x i8*] }* @_ZTVN6Kripke6TimingE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %i, align 8, !tbaa !76
  call fastcc void @_ZN6Kripke6Timing7stopAllEv(%"class.Kripke::Timing"* nonnull dereferenceable(64) %arg) #16
  %i2 = getelementptr inbounds %"class.Kripke::Timing", %"class.Kripke::Timing"* %arg, i64 0, i32 1
  %i3 = getelementptr inbounds %"class.std::map", %"class.std::map"* %i2, i64 0, i32 0
  %i4 = getelementptr inbounds %"class.std::map", %"class.std::map"* %i2, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %i5 = getelementptr inbounds i8, i8* %i4, i64 16
  %i6 = bitcast i8* %i5 to %"struct.std::_Rb_tree_node.542"**
  %i7 = load %"struct.std::_Rb_tree_node.542"*, %"struct.std::_Rb_tree_node.542"** %i6, align 8, !tbaa !192
  call fastcc void @_ZNSt8_Rb_treeINSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESt4pairIKS5_N6Kripke5TimerEESt10_Select1stISA_ESt4lessIS5_ESaISA_EE8_M_eraseEPSt13_Rb_tree_nodeISA_E(%"class.std::_Rb_tree"* nonnull dereferenceable(48) %i3, %"struct.std::_Rb_tree_node.542"* %i7) #16
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nobuiltin nounwind "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { nobuiltin allocsize(0) "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { argmemonly nounwind readonly "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { nounwind readonly }
attributes #7 = { nounwind "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #8 = { nounwind readonly "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #9 = { uwtable "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #10 = { inlinehint uwtable "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #11 = { uwtable }
attributes #12 = { nounwind uwtable }
attributes #13 = { cold noreturn nounwind }
attributes #14 = { norecurse nounwind readonly uwtable }
attributes #15 = { inlinehint uwtable }
attributes #16 = { nounwind }
attributes #17 = { builtin nounwind }
attributes #18 = { noreturn nounwind }
attributes #19 = { allocsize(0) }
attributes #20 = { nounwind readonly }

!llvm.ident = !{!44, !44, !44, !44, !44, !44, !44, !44, !44, !44, !44, !44, !44, !44, !44, !44, !44}
!nvvm.annotations = !{}
!llvm.module.flags = !{!45, !46, !47, !48, !49, !50}

!0 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageIdEE"}
!1 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EEE"}
!2 = !{i64 16, !"_ZTSN6Kripke4Core7BaseVarE"}
!3 = !{i64 16, !"_ZTSN6Kripke4Core9DomainVarE"}
!4 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneJEEEE"}
!5 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneIENS_5ZoneKEEEE"}
!6 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_5ZoneJENS_5ZoneKEEEE"}
!7 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEE"}
!8 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_5GroupENS_4ZoneEEEE"}
!9 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageINS_12GlobalSdomIdEEE"}
!10 = !{i64 16, !"_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_9DimensionEEEE"}
!11 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_9DirectionENS_6MomentEEEE"}
!12 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_6MomentENS_9DirectionEEEE"}
!13 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageIiEE"}
!14 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIiJNS_9DirectionEEEE"}
!15 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_9DirectionEEEE"}
!16 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageINS_8LegendreEEE"}
!17 = !{i64 16, !"_ZTSN6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEEE"}
!18 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_5GroupENS_4ZoneEEEE"}
!19 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageINS_7MixElemEEE"}
!20 = !{i64 16, !"_ZTSN6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEEE"}
!21 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIiJNS_4ZoneEEEE"}
!22 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_7MixElemEEEE"}
!23 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageINS_8MaterialEEE"}
!24 = !{i64 16, !"_ZTSN6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEEE"}
!25 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageINS_4ZoneEEE"}
!26 = !{i64 16, !"_ZTSN6Kripke4Core5FieldINS_4ZoneEJNS_7MixElemEEEE"}
!27 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_4ZoneEEEE"}
!28 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_5ZoneKEEEE"}
!29 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_5ZoneJEEEE"}
!30 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIdJNS_5ZoneIEEEE"}
!31 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageIlEE"}
!32 = !{i64 16, !"_ZTSN6Kripke4Core5FieldIlJNS_12GlobalSdomIdEEEE"}
!33 = !{i64 16, !"_ZTSN6Kripke4Core12FieldStorageINS_6SdomIdEEE"}
!34 = !{i64 16, !"_ZTSN6Kripke4Core5FieldINS_6SdomIdEJNS_12GlobalSdomIdEEEE"}
!35 = !{i64 16, !"_ZTSN6Kripke4Core5FieldINS_12GlobalSdomIdEJNS_6SdomIdEEEE"}
!36 = !{i64 16, !"_ZTSN6Kripke4Core3SetE"}
!37 = !{i64 32, !"_ZTSMN6Kripke4Core3SetEKFmvE.virtual"}
!38 = !{i64 40, !"_ZTSMN6Kripke4Core3SetEKFmNS_6SdomIdEmE.virtual"}
!39 = !{i64 32, !"_ZTSMN6Kripke4Core7BaseVarEKFmvE.virtual"}
!40 = !{i64 40, !"_ZTSMN6Kripke4Core7BaseVarEKFmNS_6SdomIdEmE.virtual"}
!41 = !{i64 32, !"_ZTSMN6Kripke4Core9DomainVarEKFmvE.virtual"}
!42 = !{i64 40, !"_ZTSMN6Kripke4Core9DomainVarEKFmNS_6SdomIdEmE.virtual"}
!43 = !{i64 16, !"_ZTSN6Kripke6TimingE"}
!44 = !{!"clang version 12.0.1 (git@github.com:llvm/llvm-project 4973ce53ca8abfc14233a3d8b3045673e0e8543c)"}
!45 = !{i32 1, !"wchar_size", i32 4}
!46 = !{i32 7, !"PIC Level", i32 2}
!47 = !{i32 7, !"PIE Level", i32 2}
!48 = !{i32 1, !"ThinLTO", i32 0}
!49 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!50 = !{i32 1, !"LTOPostLink", i32 1}
!51 = !{!52, !56, i64 24}
!52 = !{!"_ZTSSt18_Rb_tree_node_base", !53, i64 0, !56, i64 8, !56, i64 16, !56, i64 24}
!53 = !{!"_ZTSSt14_Rb_tree_color", !54, i64 0}
!54 = !{!"omnipotent char", !55, i64 0}
!55 = !{!"Simple C++ TBAA"}
!56 = !{!"any pointer", !54, i64 0}
!57 = !{!52, !56, i64 16}
!58 = !{!59, !56, i64 0}
!59 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE", !60, i64 0, !61, i64 8, !54, i64 16}
!60 = !{!"_ZTSNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE12_Alloc_hiderE", !56, i64 0}
!61 = !{!"long", !54, i64 0}
!62 = distinct !{!62, !63, !64}
!63 = !{!"llvm.loop.mustprogress"}
!64 = !{!"llvm.loop.unroll.disable"}
!65 = distinct !{!65, !63, !64}
!66 = distinct !{!66, !63, !64}
!67 = !{!60, !56, i64 0}
!68 = !{!54, !54, i64 0}
!69 = !{!59, !61, i64 8}
!70 = distinct !{!70, !64}
!71 = distinct !{!71, !63, !64}
!72 = distinct !{!72, !63, !64}
!73 = distinct !{!73, !64}
!74 = distinct !{!74, !63, !64}
!75 = distinct !{!75, !63, !64}
!76 = !{!77, !77, i64 0}
!77 = !{!"vtable pointer", !55, i64 0}
!78 = !{!56, !56, i64 0}
!79 = !{!80, !56, i64 0}
!80 = !{!"_ZTSSt12_Vector_baseIPdSaIS0_EE", !81, i64 0}
!81 = !{!"_ZTSNSt12_Vector_baseIPdSaIS0_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!82 = !{!83, !56, i64 0}
!83 = !{!"_ZTSSt12_Vector_baseImSaImEE", !84, i64 0}
!84 = !{!"_ZTSNSt12_Vector_baseImSaImEE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!85 = !{!86, !56, i64 0}
!86 = !{!"_ZTSSt12_Vector_baseIN6Kripke6SdomIdESaIS1_EE", !87, i64 0}
!87 = !{!"_ZTSNSt12_Vector_baseIN6Kripke6SdomIdESaIS1_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!88 = distinct !{!88, !64}
!89 = !{!90, !56, i64 0}
!90 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke8MaterialENS4_8LegendreENS4_11GlobalGroupES7_EEELln1EEESaIS9_EE", !91, i64 0}
!91 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke8MaterialENS4_8LegendreENS4_11GlobalGroupES7_EEELln1EEESaIS9_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!92 = !{!93, !56, i64 0}
!93 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_5ZoneIENS4_5ZoneJEEEELln1EEESaISA_EE", !94, i64 0}
!94 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_5ZoneIENS4_5ZoneJEEEELln1EEESaISA_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!95 = !{!96, !56, i64 0}
!96 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_5ZoneIENS4_5ZoneKEEEELln1EEESaISA_EE", !97, i64 0}
!97 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_5ZoneIENS4_5ZoneKEEEELln1EEESaISA_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!98 = !{!99, !56, i64 0}
!99 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_5ZoneJENS4_5ZoneKEEEELln1EEESaISA_EE", !100, i64 0}
!100 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_5ZoneJENS4_5ZoneKEEEELln1EEESaISA_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!101 = !{!102, !56, i64 0}
!102 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6MomentENS4_5GroupENS4_4ZoneEEEELln1EEESaIS9_EE", !103, i64 0}
!103 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6MomentENS4_5GroupENS4_4ZoneEEEELln1EEESaIS9_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!104 = !{!105, !56, i64 0}
!105 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_4ZoneEEEELln1EEESaIS9_EE", !106, i64 0}
!106 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_5GroupENS4_4ZoneEEEELln1EEESaIS9_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!107 = !{i64 0, i64 8, !108, i64 8, i64 8, !108, i64 16, i64 8, !108, i64 24, i64 8, !108, i64 32, i64 4, !110, i64 36, i64 4, !110, i64 40, i64 4, !110, i64 44, i64 4, !110}
!108 = !{!109, !109, i64 0}
!109 = !{!"double", !54, i64 0}
!110 = !{!111, !111, i64 0}
!111 = !{!"int", !54, i64 0}
!112 = distinct !{!112, !63, !64}
!113 = distinct !{!113, !63, !64}
!114 = distinct !{!114, !64}
!115 = distinct !{!115, !63, !64}
!116 = distinct !{!116, !63, !64}
!117 = distinct !{!117, !63, !64}
!118 = distinct !{!118, !64}
!119 = distinct !{!119, !63, !64}
!120 = !{!121, !56, i64 0}
!121 = !{!"_ZTSSt12_Vector_baseIPN6Kripke12GlobalSdomIdESaIS2_EE", !122, i64 0}
!122 = !{!"_ZTSNSt12_Vector_baseIPN6Kripke12GlobalSdomIdESaIS2_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!123 = distinct !{!123, !64}
!124 = !{!125, !56, i64 0}
!125 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DimensionEEEELln1EEESaIS7_EE", !126, i64 0}
!126 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DimensionEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!127 = !{!128, !56, i64 0}
!128 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_6MomentEEEELln1EEESaIS8_EE", !129, i64 0}
!129 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionENS4_6MomentEEEELln1EEESaIS8_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!130 = !{!131, !56, i64 0}
!131 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6MomentENS4_9DirectionEEEELln1EEESaIS8_EE", !132, i64 0}
!132 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6MomentENS4_9DirectionEEEELln1EEESaIS8_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!133 = !{!134, !56, i64 0}
!134 = !{!"_ZTSSt12_Vector_baseIPiSaIS0_EE", !135, i64 0}
!135 = !{!"_ZTSNSt12_Vector_baseIPiSaIS0_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!136 = distinct !{!136, !64}
!137 = !{!138, !56, i64 0}
!138 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionEEEELln1EEESaIS7_EE", !139, i64 0}
!139 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke9DirectionEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!140 = !{!141, !56, i64 0}
!141 = !{!"_ZTSSt12_Vector_baseIPN6Kripke8LegendreESaIS2_EE", !142, i64 0}
!142 = !{!"_ZTSNSt12_Vector_baseIPN6Kripke8LegendreESaIS2_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!143 = distinct !{!143, !64}
!144 = !{!145, !56, i64 0}
!145 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6MomentEEEELln1EEESaIS7_EE", !146, i64 0}
!146 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6MomentEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!147 = !{!148, !56, i64 0}
!148 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5GroupENS4_4ZoneEEEELln1EEESaIS8_EE", !149, i64 0}
!149 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5GroupENS4_4ZoneEEEELln1EEESaIS8_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!150 = !{!151, !56, i64 0}
!151 = !{!"_ZTSSt12_Vector_baseIPN6Kripke7MixElemESaIS2_EE", !152, i64 0}
!152 = !{!"_ZTSNSt12_Vector_baseIPN6Kripke7MixElemESaIS2_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!153 = distinct !{!153, !64}
!154 = !{!155, !56, i64 0}
!155 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke4ZoneEEEELln1EEESaIS7_EE", !156, i64 0}
!156 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke4ZoneEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!157 = !{!158, !56, i64 0}
!158 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke7MixElemEEEELln1EEESaIS7_EE", !159, i64 0}
!159 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke7MixElemEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!160 = !{!161, !56, i64 0}
!161 = !{!"_ZTSSt12_Vector_baseIPN6Kripke8MaterialESaIS2_EE", !162, i64 0}
!162 = !{!"_ZTSNSt12_Vector_baseIPN6Kripke8MaterialESaIS2_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!163 = distinct !{!163, !64}
!164 = !{!165, !56, i64 0}
!165 = !{!"_ZTSSt12_Vector_baseIPN6Kripke4ZoneESaIS2_EE", !166, i64 0}
!166 = !{!"_ZTSNSt12_Vector_baseIPN6Kripke4ZoneESaIS2_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!167 = distinct !{!167, !64}
!168 = !{!169, !56, i64 0}
!169 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5ZoneKEEEELln1EEESaIS7_EE", !170, i64 0}
!170 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5ZoneKEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!171 = !{!172, !56, i64 0}
!172 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5ZoneJEEEELln1EEESaIS7_EE", !173, i64 0}
!173 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5ZoneJEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!174 = !{!175, !56, i64 0}
!175 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5ZoneIEEEELln1EEESaIS7_EE", !176, i64 0}
!176 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke5ZoneIEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!177 = !{!178, !56, i64 0}
!178 = !{!"_ZTSSt12_Vector_baseIPlSaIS0_EE", !179, i64 0}
!179 = !{!"_ZTSNSt12_Vector_baseIPlSaIS0_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!180 = distinct !{!180, !64}
!181 = !{!182, !56, i64 0}
!182 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke12GlobalSdomIdEEEELln1EEESaIS7_EE", !183, i64 0}
!183 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke12GlobalSdomIdEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!184 = !{!185, !56, i64 0}
!185 = !{!"_ZTSSt12_Vector_baseIPN6Kripke6SdomIdESaIS2_EE", !186, i64 0}
!186 = !{!"_ZTSNSt12_Vector_baseIPN6Kripke6SdomIdESaIS2_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!187 = distinct !{!187, !64}
!188 = !{!189, !56, i64 0}
!189 = !{!"_ZTSSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6SdomIdEEEELln1EEESaIS7_EE", !190, i64 0}
!190 = !{!"_ZTSNSt12_Vector_baseIN4RAJA11TypedLayoutIlN4camp5tupleIJN6Kripke6SdomIdEEEELln1EEESaIS7_EE12_Vector_implE", !56, i64 0, !56, i64 8, !56, i64 16}
!191 = !{!61, !61, i64 0}
!192 = !{!193, !56, i64 8}
!193 = !{!"_ZTSSt15_Rb_tree_header", !52, i64 0, !61, i64 32}
!194 = !{!195, !56, i64 32}
!195 = !{!"_ZTSSt4pairIKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPN6Kripke4Core7BaseVarEE", !59, i64 0, !56, i64 32}
!196 = !{i64 0, i64 4, !197, i64 4, i64 4, !199}
!197 = !{!198, !198, i64 0}
!198 = !{!"_ZTSN6Kripke5ArchVE", !54, i64 0}
!199 = !{!200, !200, i64 0}
!200 = !{!"_ZTSN6Kripke7LayoutVE", !54, i64 0}
!201 = distinct !{!201, !63, !64}
!202 = distinct !{!202, !64}
!203 = !{!204, !56, i64 8}
!204 = !{!"_ZTSZN6Kripke8dispatchI14ScatteringSdomJRNS_6SdomIdES3_RNS_4Core3SetES6_S6_RNS4_5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEESC_RNS7_IdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESF_EEERNS7_INS_7MixElemEJSA_EEERNS7_IiJSA_EEERNS7_ISD_JSI_EEERNS7_IdJSI_EEERNS7_ISE_JS8_EEEEEEvNS_11ArchLayoutVERKT_DpOT0_EUlSU_E_", !56, i64 0, !56, i64 8, !56, i64 16, !56, i64 24, !56, i64 32, !56, i64 40, !56, i64 48, !56, i64 56, !56, i64 64, !56, i64 72, !56, i64 80, !56, i64 88, !56, i64 96, !56, i64 104, !56, i64 112}
!205 = !{!204, !56, i64 16}
!206 = !{!204, !56, i64 24}
!207 = !{!204, !56, i64 32}
!208 = !{!204, !56, i64 40}
!209 = !{!204, !56, i64 48}
!210 = !{!204, !56, i64 56}
!211 = !{!204, !56, i64 64}
!212 = !{!204, !56, i64 72}
!213 = !{!204, !56, i64 80}
!214 = !{!204, !56, i64 88}
!215 = !{!204, !56, i64 96}
!216 = !{!204, !56, i64 104}
!217 = !{!204, !56, i64 112}
!218 = !{!219, !221}
!219 = distinct !{!219, !220, !"_ZNK6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionES2_S3_NS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEEN4RAJA4ViewIS2_NS0_10LayoutInfoIT_JS3_EE6LayoutEPS2_EENS_6SdomIdE: %agg.result"}
!220 = distinct !{!220, !"_ZNK6Kripke4Core5FieldINS_8LegendreEJNS_6MomentEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionES2_S3_NS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEEN4RAJA4ViewIS2_NS0_10LayoutInfoIT_JS3_EE6LayoutEPS2_EENS_6SdomIdE"}
!221 = distinct !{!221, !222, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldINS_8LegendreEJNS_6MomentEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionES9_SA_NS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEdtdefpT7sdom_idEERT_: %agg.result"}
!222 = distinct !{!222, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldINS_8LegendreEJNS_6MomentEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionES9_SA_NS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEdtdefpT7sdom_idEERT_"}
!223 = !{!224, !226}
!224 = distinct !{!224, !225, !"_ZNK6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES2_S4_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupES3_NS_7MixElemEEEEEEN4RAJA4ViewIdNS0_10LayoutInfoIT_JS2_S3_S4_EE6LayoutEPdEENS_6SdomIdE: %agg.result"}
!225 = distinct !{!225, !"_ZNK6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES2_S4_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupES3_NS_7MixElemEEEEEEN4RAJA4ViewIdNS0_10LayoutInfoIT_JS2_S3_S4_EE6LayoutEPdEENS_6SdomIdE"}
!226 = distinct !{!226, !227, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES9_SB_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupESA_NS_7MixElemEEEEEdtdefpT7sdom_idEERT_: %agg.result"}
!227 = distinct !{!227, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES9_SB_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupESA_NS_7MixElemEEEEEdtdefpT7sdom_idEERT_"}
!228 = !{!229, !231}
!229 = distinct !{!229, !230, !"_ZNK6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES2_S4_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupES3_NS_7MixElemEEEEEEN4RAJA4ViewIdNS0_10LayoutInfoIT_JS2_S3_S4_EE6LayoutEPdEENS_6SdomIdE: %agg.result"}
!230 = distinct !{!230, !"_ZNK6Kripke4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES2_S4_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupES3_NS_7MixElemEEEEEEN4RAJA4ViewIdNS0_10LayoutInfoIT_JS2_S3_S4_EE6LayoutEPdEENS_6SdomIdE"}
!231 = distinct !{!231, !232, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES9_SB_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupESA_NS_7MixElemEEEEEfp0_EERT_NS_6SdomIdE: %agg.result"}
!232 = distinct !{!232, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIdJNS_6MomentENS_5GroupENS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreES9_SB_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupESA_NS_7MixElemEEEEEfp0_EERT_NS_6SdomIdE"}
!233 = !{!234, !236}
!234 = distinct !{!234, !235, !"_ZNK6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EE12getViewOrderIN4camp4listIJlNS_9DimensionES2_NS_9DirectionES3_NS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIES4_NS_5GroupENS_7MixElemEEEEEEN4RAJA4ViewIdNS0_10LayoutInfoIT_JS2_S3_S4_S4_EE6LayoutEPdEENS_6SdomIdE: %agg.result"}
!235 = distinct !{!235, !"_ZNK6Kripke4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupES4_EE12getViewOrderIN4camp4listIJlNS_9DimensionES2_NS_9DirectionES3_NS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIES4_NS_5GroupENS_7MixElemEEEEEEN4RAJA4ViewIdNS0_10LayoutInfoIT_JS2_S3_S4_S4_EE6LayoutEPdEENS_6SdomIdE"}
!236 = distinct !{!236, !237, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESB_EEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionES9_NS_9DirectionESA_NS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIESB_NS_5GroupENS_7MixElemEEEEEdtdefpT7sdom_idEERT_: %agg.result"}
!237 = distinct !{!237, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIdJNS_8MaterialENS_8LegendreENS_11GlobalGroupESB_EEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionES9_NS_9DirectionESA_NS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIESB_NS_5GroupENS_7MixElemEEEEEdtdefpT7sdom_idEERT_"}
!238 = !{!239, !241}
!239 = distinct !{!239, !240, !"_ZNK6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentES3_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupES2_EEEEEN4RAJA4ViewIS2_NS0_10LayoutInfoIT_JS3_EE6LayoutEPS2_EENS_6SdomIdE: %agg.result"}
!240 = distinct !{!240, !"_ZNK6Kripke4Core5FieldINS_7MixElemEJNS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentES3_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupES2_EEEEEN4RAJA4ViewIS2_NS0_10LayoutInfoIT_JS3_EE6LayoutEPS2_EENS_6SdomIdE"}
!241 = distinct !{!241, !242, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldINS_7MixElemEJNS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentESA_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupES9_EEEEdtdefpT7sdom_idEERT_: %agg.result"}
!242 = distinct !{!242, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldINS_7MixElemEJNS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentESA_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupES9_EEEEdtdefpT7sdom_idEERT_"}
!243 = !{!244, !246}
!244 = distinct !{!244, !245, !"_ZNK6Kripke4Core5FieldIiJNS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentES2_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEEN4RAJA4ViewIiNS0_10LayoutInfoIT_JS2_EE6LayoutEPiEENS_6SdomIdE: %agg.result"}
!245 = distinct !{!245, !"_ZNK6Kripke4Core5FieldIiJNS_4ZoneEEE12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentES2_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEEN4RAJA4ViewIiNS0_10LayoutInfoIT_JS2_EE6LayoutEPiEENS_6SdomIdE"}
!246 = distinct !{!246, !247, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIiJNS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentES9_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEdtdefpT7sdom_idEERT_: %agg.result"}
!247 = distinct !{!247, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldIiJNS_4ZoneEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionENS_8MaterialENS_9DirectionENS_8LegendreENS_6MomentES9_NS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupENS_7MixElemEEEEEdtdefpT7sdom_idEERT_"}
!248 = !{!249, !251}
!249 = distinct !{!249, !250, !"_ZNK6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEE12getViewOrderIN4camp4listIJlNS_9DimensionES2_NS_9DirectionENS_8LegendreENS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupES3_EEEEEN4RAJA4ViewIS2_NS0_10LayoutInfoIT_JS3_EE6LayoutEPS2_EENS_6SdomIdE: %agg.result"}
!250 = distinct !{!250, !"_ZNK6Kripke4Core5FieldINS_8MaterialEJNS_7MixElemEEE12getViewOrderIN4camp4listIJlNS_9DimensionES2_NS_9DirectionENS_8LegendreENS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupES3_EEEEEN4RAJA4ViewIS2_NS0_10LayoutInfoIT_JS3_EE6LayoutEPS2_EENS_6SdomIdE"}
!251 = distinct !{!251, !252, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldINS_8MaterialEJNS_7MixElemEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionES9_NS_9DirectionENS_8LegendreENS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupESA_EEEEdtdefpT7sdom_idEERT_: %agg.result"}
!252 = distinct !{!252, !"_ZNK6Kripke6SdomALINS_11ArchLayoutTINS_16ArchT_SequentialENS_11LayoutT_DZGEEEE7getViewINS_4Core5FieldINS_8MaterialEJNS_7MixElemEEEEEEDTcldtfp_12getViewOrderIN4camp4listIJlNS_9DimensionES9_NS_9DirectionENS_8LegendreENS_6MomentENS_4ZoneENS_5ZoneKENS_5ZoneJENS_5ZoneIENS_11GlobalGroupENS_5GroupESA_EEEEdtdefpT7sdom_idEERT_"}
!253 = distinct !{!253, !63, !64}
!254 = distinct !{!254, !63, !64}
!255 = distinct !{!255, !63, !64}
!256 = distinct !{!256, !63, !64}
!257 = distinct !{!257, !63, !64}
!258 = !{i64 0, i64 8, !191}
!259 = !{!260, !109, i64 16}
!260 = !{!"_ZTSN4RAJA11ChronoTimerE", !261, i64 0, !261, i64 8, !109, i64 16}
!261 = !{!"_ZTSNSt6chrono10time_pointINS_3_V212steady_clockENS_8durationIlSt5ratioILl1ELl1000000000EEEEEE", !262, i64 0}
!262 = !{!"_ZTSNSt6chrono8durationIlSt5ratioILl1ELl1000000000EEEE", !61, i64 0}
!263 = !{!264, !265, i64 0}
!264 = !{!"_ZTSN6Kripke5TimerE", !265, i64 0, !109, i64 8, !61, i64 16, !266, i64 24}
!265 = !{!"bool", !54, i64 0}
!266 = !{!"_ZTSN4RAJA5TimerE"}
!267 = !{!264, !61, i64 16}
!268 = distinct !{!268, !63, !64}
!269 = !{!193, !61, i64 32}
!270 = !{!271, !56, i64 0}
!271 = !{!"_ZTSSt10_Head_baseILm0ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEELb0EE", !56, i64 0}
!272 = distinct !{!272, !63, !64}
!273 = !{!193, !56, i64 16}
!274 = !{i8 0, i8 2}
!275 = !{!264, !109, i64 8}
!276 = distinct !{!276, !64}
!277 = distinct !{!277, !63, !64}

; CHECK: @diffe_ZN6Kripke6Kernel10scatteringERNS_4Core9DataStoreE
