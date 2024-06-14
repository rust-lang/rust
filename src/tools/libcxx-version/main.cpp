// Detecting the standard library version manually using a bunch of shell commands is very
// complicated and fragile across different platforms. This program provides the major version
// of the standard library on any target platform without requiring any messy work.
//
// It's nothing more than specifying the name of the standard library implementation (either libstdc++ or libc++)
// and its major version.

#include <cstdio>

int main() {
    #ifdef _GLIBCXX_RELEASE
        #define name "libstdc++"
        #define version _GLIBCXX_RELEASE
    #elif defined(_LIBCPP_VERSION)
        // _LIBCPP_VERSION follows "XXYYZZ" format (e.g., 170001 for 17.0.1).
        // ref: https://github.com/llvm/llvm-project/blob/f64732195c1030ee2627ff4e4142038e01df1d26/libcxx/include/__config#L51-L54
        //
        // Since we use the major version from _GLIBCXX_RELEASE, we need to extract only the first 2 characters of _LIBCPP_VERSION
        // to provide the major version for consistency.
        #define name "libc++"
        #define version _LIBCPP_VERSION / 10000
    #else
        #error "Couldn't recognize the standard library version."
    #endif

    printf("%s version: %d\n", name, version);

    return 0;
}
