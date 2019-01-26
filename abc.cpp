// compile using g++ -std=c++11 -g -c abc.cpp -o abc.o

struct Opaque;

struct MyType {
    unsigned int field_a;
    int field_b;
    void* field_c;
    float field_d;
    //double field_e;
    //long long field_f;
    bool field_g;
    char field_h;
    Opaque* field_i;
};

MyType bcd(int x, MyType a) {
    MyType b = a;
    MyType c = b;
    MyType d = c;
    MyType e = d;
    MyType f = e;
    MyType g = f;
    MyType h = g;
    MyType i = h;
    MyType j = i;
    MyType k = j;
    MyType l = k;
    MyType m = l;
    return b;
}
int main() {
    bcd(42, {});
    return 0;
}
