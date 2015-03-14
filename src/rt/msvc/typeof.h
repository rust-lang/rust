// This piece of magic brought to you by:
//     http://www.nedproductions.biz/blog/
//     implementing-typeof-in-microsofts-c-compiler 

#ifndef MSVC_TYPEOF_H
#define MSVC_TYPEOF_H

#if defined(_MSC_VER) && _MSC_VER>=1400
namespace msvc_typeof_impl {
	/* This is a fusion of Igor Chesnokov's method (http://rsdn.ru/forum/src/1094305.aspx)
	and Steven Watanabe's method (http://lists.boost.org/Archives/boost/2006/12/115006.php)

	How it works:
	C++ allows template type inference for templated function parameters but nothing else.
	What we do is to pass the expression sent to typeof() into the templated function vartypeID()
	as its parameter, thus extracting its type. The big problem traditionally now is how to get
	that type out of the vartypeID() instance, and here's how we do it:
		1. unique_type_id() returns a monotonically increasing integer for every unique type
		   passed to it during this compilation unit. It also specialises an instance of
		   msvc_extract_type<unique_type_id, type>::id2type_impl<true>.
		2. vartypeID() returns a sized<unique_type_id> for the type where
		   sizeof(sized<unique_type_id>)==unique_type_id. We vector through sized as a means
		   of returning the unique_type_id at compile time rather than runtime.
		3. msvc_extract_type<unique_type_id> then extracts the type by using a bug in MSVC to
		   reselect the specialised child type (id2type_impl<true>) from within the specialisation
		   of itself originally performed by the above instance of unique_type_id. This bug works
		   because when MSVC calculated the signature of the specialised
		   msvc_extract_type<unique_type_id, type>::id2type_impl<true>, it does not include the
		   value of type in the signature of id2type_impl<true>. Therefore when we reselect
		   msvc_extract_type<unique_type_id>::id2type_impl<true> it erroneously returns the one
		   already in its list of instantiated types rather than correctly generating a newly
		   specialised msvc_extract_type<unique_type_id, msvc_extract_type_default_param>::id2type_impl<true>

	This bug allows the impossible and gives us a working typeof() in MSVC. Hopefully Microsoft
	won't fix this bug until they implement a native typeof.
	*/

	struct msvc_extract_type_default_param {};
	template<int ID, typename T = msvc_extract_type_default_param> struct msvc_extract_type;

	template<int ID> struct msvc_extract_type<ID, msvc_extract_type_default_param>
	{
		template<bool> struct id2type_impl; 

		typedef id2type_impl<true> id2type; 
	};

	template<int ID, typename T> struct msvc_extract_type : msvc_extract_type<ID, msvc_extract_type_default_param> 
	{ 
		template<> struct id2type_impl<true> //VC8.0 specific bugfeature 
		{ 
			typedef T type; 
		}; 
		template<bool> struct id2type_impl; 

		typedef id2type_impl<true> id2type; 
	}; 


	template<int N> class CCounter;

	// TUnused is required to force compiler to recompile CCountOf class
	template<typename TUnused, int NTested = 0> struct CCountOf
	{
		enum
		{
			__if_exists(CCounter<NTested>) { count = CCountOf<TUnused, NTested + 1>::count }
			__if_not_exists(CCounter<NTested>) { count = NTested }
		};
	};

	template<class TTypeReg, class TUnused, int NValue> struct CProvideCounterValue { enum { value = NValue }; };

	// type_id
	#define unique_type_id(type) \
		(CProvideCounterValue< \
			/*register TYPE--ID*/ typename msvc_extract_type<CCountOf<type >::count, type>::id2type, \
			/*increment compile-time Counter*/ CCounter<CCountOf<type >::count>, \
			/*pass value of Counter*/CCountOf<type >::count \
		 >::value)

	// Lets type_id() be > than 0
	class __Increment_type_id { enum { value = unique_type_id(__Increment_type_id) }; };

	// vartypeID() returns a type with sizeof(type_id)
	template<int NSize>	class sized { char m_pad[NSize]; };
	template<typename T> typename sized<unique_type_id(T)> vartypeID(T&);
	template<typename T> typename sized<unique_type_id(const T)> vartypeID(const T&);
	template<typename T> typename sized<unique_type_id(volatile  T)> vartypeID(volatile T&);
	template<typename T> typename sized<unique_type_id(const volatile T)> vartypeID(const volatile T&);
}

#define typeof(expression) msvc_typeof_impl::msvc_extract_type<sizeof(msvc_typeof_impl::vartypeID(expression))>::id2type::type
#endif

#endif
