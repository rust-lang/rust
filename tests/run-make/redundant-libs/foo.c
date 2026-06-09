#ifdef _MSC_VER
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

DllExport void foo1() {}
DllExport void foo2() {}
